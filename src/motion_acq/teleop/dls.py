"""Incremental damped-least-squares IK for real-time teleoperation.

This is the local HandUMI counterpart of Interlatent's browser-side
``DlsSolver``.  It deliberately performs one warm-started Newton step per
tracking frame instead of globally re-solving IK.  The existing HandUMI
clutch, One-Euro filter, trajectory interpolation, and hardware safety layers
remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

# Real hardware may advertise several rad/s. Directly using that mechanical
# ceiling for human teleoperation is needlessly aggressive, so DLS intersects
# the discovered URDF/backend limits with one conservative system policy.
DEFAULT_TELEOP_JOINT_SPEED_CAP_RAD_S = 1.0
DEFAULT_DLS_TRACKING_RATE_HZ = 72.0
DEFAULT_LM_TRACKING_RATE_HZ = 30.0
TRAJECTORY_SCHEDULING_MARGIN_MS = 1000.0 / 150.0


def resolve_real_teleop_timing(
    ik_solver: str,
    *,
    input_rate_hz: float | None,
    trajectory_delay_ms: float | None,
) -> tuple[float, float]:
    """Resolve the shared live-control cadence for real and recording modes."""
    rate_hz = (
        float(input_rate_hz)
        if input_rate_hz is not None
        else (
            DEFAULT_DLS_TRACKING_RATE_HZ
            if ik_solver == "dls"
            else DEFAULT_LM_TRACKING_RATE_HZ
        )
    )
    delay_ms = (
        float(trajectory_delay_ms)
        if trajectory_delay_ms is not None
        else 1000.0 / rate_hz + TRAJECTORY_SCHEDULING_MARGIN_MS
    )
    return rate_hz, delay_ms


@dataclass(frozen=True)
class DlsConfig:
    """Weighted-DLS parameters matching Interlatent's canonical defaults."""

    lambda_position: float = 0.05
    lambda_singularity: float = 0.15
    singular_value_threshold: float = 0.05
    rest_gain: float = 0.02
    orientation_weight: float = 0.1
    rotation_error_hold_rad: float = 2.2

    def validate(self) -> None:
        values = (
            self.lambda_position,
            self.lambda_singularity,
            self.singular_value_threshold,
            self.rest_gain,
            self.orientation_weight,
            self.rotation_error_hold_rad,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("DLS parameters must be finite")
        if self.lambda_position <= 0.0:
            raise ValueError("lambda_position must be > 0")
        if self.lambda_singularity < 0.0:
            raise ValueError("lambda_singularity must be >= 0")
        if self.singular_value_threshold <= 0.0:
            raise ValueError("singular_value_threshold must be > 0")
        if self.rest_gain < 0.0:
            raise ValueError("rest_gain must be >= 0")
        if self.orientation_weight < 0.0:
            raise ValueError("orientation_weight must be >= 0")
        if self.rotation_error_hold_rad < 0.0:
            raise ValueError("rotation_error_hold_rad must be >= 0")


class IncrementalDlsSolver:
    """One-step, per-arm DLS adapter for ``BimanualKinematicsSolver``.

    The adapter preserves the solver interface consumed by
    :class:`~motion_acq.teleop.core.TeleopController`.  FK is delegated to the
    original solver while IK uses differentiable Pyroki FK to form a local
    Jacobian at the previous joint command.
    """

    def __init__(
        self,
        base_solver,
        *,
        side_joint_indices: dict[str, tuple[int, ...]],
        side_joint_speed_limits_rad_s: dict[str, np.ndarray],
        rest_q: np.ndarray,
        nominal_rate_hz: float,
        config: DlsConfig | None = None,
    ) -> None:
        self.base_solver = base_solver
        self.robot = base_solver.robot
        self.ee_indices = base_solver.ee_indices
        self.rest_q = np.asarray(rest_q, dtype=np.float32).copy()
        if nominal_rate_hz <= 0.0:
            raise ValueError("nominal_rate_hz must be > 0")
        self.nominal_dt_s = 1.0 / float(nominal_rate_hz)
        self._step_dt_s = self.nominal_dt_s
        self.config = config or DlsConfig()
        self.config.validate()

        joint_count = self.robot.joints.num_actuated_joints
        if self.rest_q.shape != (joint_count,):
            raise ValueError(
                f"rest_q has shape {self.rest_q.shape}, expected {(joint_count,)}"
            )
        lower = np.asarray(self.robot.joints.lower_limits, dtype=np.float32)
        upper = np.asarray(self.robot.joints.upper_limits, dtype=np.float32)
        if lower.shape != (joint_count,) or upper.shape != (joint_count,):
            raise ValueError("robot joint limits do not match its actuated joints")

        locked = set(getattr(base_solver, "locked_joint_indices", ()))
        self.side_joint_indices: dict[str, tuple[int, ...]] = {}
        self.side_joint_speed_limits_rad_s: dict[str, np.ndarray] = {}
        self._steps = {}
        for side, ee_index in zip(("left", "right"), self.ee_indices, strict=True):
            indices = tuple(
                index
                for index in side_joint_indices[side]
                if index not in locked
            )
            if not indices:
                raise ValueError(f"DLS has no movable joints for the {side} arm")
            speed_limits = np.asarray(
                side_joint_speed_limits_rad_s[side], dtype=np.float32
            )
            if speed_limits.shape != (len(indices),):
                raise ValueError(
                    f"DLS {side} speed limits have shape {speed_limits.shape}, "
                    f"expected {(len(indices),)}"
                )
            if not np.all(np.isfinite(speed_limits)) or np.any(speed_limits <= 0.0):
                raise ValueError(f"DLS {side} speed limits must be finite and > 0")
            self.side_joint_indices[side] = indices
            self.side_joint_speed_limits_rad_s[side] = speed_limits
            self._steps[side] = self._make_step(
                ee_index=ee_index,
                indices=indices,
                lower=lower[np.asarray(indices, dtype=np.intp)],
                upper=upper[np.asarray(indices, dtype=np.intp)],
            )

    def __getattr__(self, name):
        """Keep non-IK compatibility with the wrapped kinematics solver."""
        return getattr(self.base_solver, name)

    def _make_step(
        self,
        *,
        ee_index: int,
        indices: tuple[int, ...],
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        robot = self.robot
        cfg = self.config
        index_array = jnp.asarray(indices, dtype=jnp.int32)
        lower_array = jnp.asarray(lower, dtype=jnp.float32)
        upper_array = jnp.asarray(upper, dtype=jnp.float32)

        def step(q_full, target_pos, target_xyzw, rest_full, max_delta):
            q_side = q_full[index_array]
            target_wxyz = target_xyzw[jnp.asarray((3, 0, 1, 2))]

            def error(candidate_side):
                candidate = q_full.at[index_array].set(candidate_side)
                fk = robot.forward_kinematics(candidate)
                current = jaxlie.SE3(fk[ee_index])
                position_error = target_pos - current.translation()
                rotation_error = (
                    jaxlie.SO3(target_wxyz) @ current.rotation().inverse()
                ).log()
                return jnp.concatenate((position_error, rotation_error))

            task_error = error(q_side)
            # ``error`` shrinks as q advances, hence the task Jacobian is the
            # negative derivative of the remaining error.
            jacobian = -jax.jacfwd(error)(q_side)
            include_rotation = (
                jnp.linalg.norm(task_error[3:]) <= cfg.rotation_error_hold_rad
            )
            weights = jnp.concatenate(
                (
                    jnp.ones(3, dtype=jnp.float32),
                    jnp.full(
                        3,
                        jnp.where(include_rotation, cfg.orientation_weight, 0.0),
                        dtype=jnp.float32,
                    ),
                )
            )
            weighted_jacobian = jacobian * weights[:, None]
            weighted_error = task_error * weights

            singular_values = jnp.linalg.svd(
                weighted_jacobian, compute_uv=False, full_matrices=False
            )
            sigma_min = jnp.min(singular_values)
            ramp = jnp.maximum(
                0.0, 1.0 - sigma_min / cfg.singular_value_threshold
            )
            lambda_squared = (
                cfg.lambda_position**2
                + cfg.lambda_singularity**2 * ramp**2
            )
            mu_squared = cfg.rest_gain**2
            normal = weighted_jacobian.T @ weighted_jacobian
            normal += (lambda_squared + mu_squared) * jnp.eye(
                len(indices), dtype=jnp.float32
            )
            rhs = weighted_jacobian.T @ weighted_error
            rhs += mu_squared * (rest_full[index_array] - q_side)
            delta = jnp.linalg.solve(normal, rhs)

            candidate = jnp.clip(q_side + delta, lower_array, upper_array)
            bounded_delta = candidate - q_side
            bounded_delta = jnp.clip(bounded_delta, -max_delta, max_delta)
            return jnp.clip(q_side + bounded_delta, lower_array, upper_array)

        return jax.jit(step)

    def warmup(self, q: np.ndarray) -> None:
        """Compile both fixed-shape arm steps before hardware is connected."""
        left, right = self.fk_pose7(q)
        self.ik(
            q,
            left_pose=(left[:3], left[3:7]),
            right_pose=(right[:3], right[3:7]),
        )

    def set_timestep(self, dt_s: float) -> None:
        """Set elapsed source-frame time used by the joint velocity clamp.

        A two-frame ceiling avoids turning a tracking stall into a large
        catch-up step. Stale tracking is independently rejected by teleop-real.
        """
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = self.nominal_dt_s
        self._step_dt_s = min(float(dt_s), 2.0 * self.nominal_dt_s)

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance each requested arm by exactly one weighted-DLS step."""
        del left_elbow_pos, right_elbow_pos
        q = np.asarray(q_current, dtype=np.float32).copy()
        for side, pose in (("left", left_pose), ("right", right_pose)):
            if pose is None:
                continue
            position, quaternion_xyzw = pose
            q_side = self._steps[side](
                jnp.asarray(q),
                jnp.asarray(position, dtype=jnp.float32),
                jnp.asarray(quaternion_xyzw, dtype=jnp.float32),
                jnp.asarray(self.rest_q),
                jnp.asarray(
                    self.side_joint_speed_limits_rad_s[side] * self._step_dt_s,
                    dtype=jnp.float32,
                ),
            )
            q[np.asarray(self.side_joint_indices[side], dtype=np.intp)] = np.asarray(
                q_side, dtype=np.float32
            )
        return self.base_solver._with_locked_joints(q)


def make_real_teleop_dls_solver(
    runtime,
    base_solver,
    home_q: np.ndarray,
    *,
    input_rate_hz: float = 30.0,
):
    """Build the DLS adapter while excluding fingers and locked joints."""
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    side_joint_indices = {
        side: tuple(
            index
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        )
        for side in ("left", "right")
    }
    backend_limit = _backend_joint_speed_limit_rad_s(runtime)
    side_speed_limits = {}
    urdf_limits = np.asarray(runtime.robot.joints.velocity_limits, dtype=np.float32)
    for side, indices in side_joint_indices.items():
        discovered = urdf_limits[np.asarray(indices, dtype=np.intp)]
        side_speed_limits[side] = np.minimum(
            discovered,
            min(backend_limit, DEFAULT_TELEOP_JOINT_SPEED_CAP_RAD_S),
        ).astype(np.float32)
    return IncrementalDlsSolver(
        base_solver,
        side_joint_indices=side_joint_indices,
        side_joint_speed_limits_rad_s=side_speed_limits,
        rest_q=home_q,
        nominal_rate_hz=input_rate_hz,
    )


def _backend_joint_speed_limit_rad_s(runtime) -> float:
    """Read the backend's portable speed ceiling in SI units."""
    control = runtime.config.real_options.get("control") or {}
    if control.get("max_joint_speed_rad_s") is not None:
        value = float(control["max_joint_speed_rad_s"])
    else:
        value = float(np.deg2rad(runtime.config.real.max_joint_speed_deg_s))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("real backend joint speed limit must be finite and > 0")
    return value


__all__ = [
    "DEFAULT_DLS_TRACKING_RATE_HZ",
    "DEFAULT_LM_TRACKING_RATE_HZ",
    "DEFAULT_TELEOP_JOINT_SPEED_CAP_RAD_S",
    "DlsConfig",
    "IncrementalDlsSolver",
    "TRAJECTORY_SCHEDULING_MARGIN_MS",
    "make_real_teleop_dls_solver",
    "resolve_real_teleop_timing",
]
