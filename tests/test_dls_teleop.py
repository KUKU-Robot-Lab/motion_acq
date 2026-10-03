"""Safety and continuity checks for the real-teleop DLS follower."""

import numpy as np
import pytest

from motion_acq.robots.registry import load_embodiment
from motion_acq.teleop.dls import (
    DlsConfig,
    IncrementalDlsSolver,
    make_real_teleop_dls_solver,
)


@pytest.fixture(scope="module")
def openarm_dls():
    runtime = load_embodiment("openarmv1")
    home_q = runtime.home_q()
    solver = make_real_teleop_dls_solver(runtime, runtime.solver_cls(), home_q)
    solver.warmup(home_q)
    return runtime, solver, home_q


def test_dls_step_is_bounded_and_does_not_move_inactive_arm(openarm_dls) -> None:
    runtime, solver, home_q = openarm_dls
    solver.set_timestep(1.0 / 30.0)
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[:3] += np.array((0.08, -0.04, 0.03), dtype=np.float32)

    actual = solver.ik(home_q, left_pose=(target[:3], target[3:7]))

    left_indices = np.asarray(solver.side_joint_indices["left"], dtype=np.intp)
    right_indices = np.asarray(runtime.arm_joint_indices("right"), dtype=np.intp)
    speed_limits = solver.side_joint_speed_limits_rad_s["left"]
    assert np.all(
        np.abs(actual[left_indices] - home_q[left_indices])
        <= speed_limits / 30.0 + 1e-7
    )
    np.testing.assert_array_equal(actual[right_indices], home_q[right_indices])
    assert np.all(actual >= np.asarray(runtime.robot.joints.lower_limits) - 1e-7)
    assert np.all(actual <= np.asarray(runtime.robot.joints.upper_limits) + 1e-7)


def _converge(solver, home_q, *, offset_m: float = 0.02, steps: int = 60):
    solver.set_timestep(1.0 / 30.0)
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[0] += offset_m
    q = home_q.copy()
    errors = []
    for _ in range(steps):
        q = solver.ik(q, left_pose=(target[:3], target[3:7]))
        achieved, _ = solver.fk_pose7(q)
        errors.append(float(np.linalg.norm(achieved[:3] - target[:3])))
    return q, errors


def test_dls_repeated_steps_reduce_cartesian_error(openarm_dls) -> None:
    """The follower closes most of the gap and then holds a steady offset.

    It does not converge to zero, and should not be expected to: this is a
    weighted solver whose orientation-hold and rest-posture terms pull against
    the position task, so it settles at their equilibrium. See the test below,
    which pins that this offset is the trade-off and not a solver defect.
    """
    _, solver, home_q = openarm_dls
    q, errors = _converge(solver, home_q)

    initial_error = 0.02
    assert errors[19] < initial_error * 0.25  # most of the gap within 20 steps
    assert errors[-1] < 0.005  # settles within 5 mm
    assert errors[-1] <= errors[19] + 1e-6  # settles instead of drifting away
    assert np.isfinite(q).all()


def test_dls_steady_state_offset_is_the_secondary_objectives(openarm_dls) -> None:
    """Disabling orientation hold and rest posture converges to the target.

    Guards the actual invariant: the damped-least-squares core is exact, and
    the residual millimeters come from the weights, so a regression that makes
    the solver itself inexact still fails here.
    """
    runtime, reference, home_q = openarm_dls
    solver = IncrementalDlsSolver(
        runtime.solver_cls(),
        side_joint_indices=reference.side_joint_indices,
        side_joint_speed_limits_rad_s=reference.side_joint_speed_limits_rad_s,
        rest_q=home_q,
        nominal_rate_hz=30.0,
        config=DlsConfig(orientation_weight=0.0, rest_gain=0.0),
    )
    solver.warmup(home_q)
    _, errors = _converge(solver, home_q)

    assert errors[-1] < 1e-4
    assert errors[-1] <= max(0.05 * errors[19], 1e-6)


def test_dls_joint_cap_scales_with_source_frame_time() -> None:
    runtime = load_embodiment("openarmv1")
    home_q = runtime.home_q()
    solver = make_real_teleop_dls_solver(
        runtime,
        runtime.solver_cls(),
        home_q,
        input_rate_hz=72.0,
    )
    left_pose, _ = solver.fk_pose7(home_q)
    target = left_pose.copy()
    target[:3] += np.array((0.20, -0.10, 0.10), dtype=np.float32)
    solver.set_timestep(1.0 / 72.0)

    actual = solver.ik(home_q, left_pose=(target[:3], target[3:7]))

    indices = np.asarray(solver.side_joint_indices["left"], dtype=np.intp)
    speeds = solver.side_joint_speed_limits_rad_s["left"]
    assert np.all(np.abs(actual[indices] - home_q[indices]) <= speeds / 72.0 + 1e-7)


def test_dls_speed_limits_are_derived_from_robot_and_backend() -> None:
    for robot_name in ("openarmv1",):
        runtime = load_embodiment(robot_name)
        solver = make_real_teleop_dls_solver(
            runtime,
            runtime.solver_cls(),
            runtime.home_q(),
            input_rate_hz=72.0,
        )
        urdf = np.asarray(runtime.robot.joints.velocity_limits)
        for side in ("left", "right"):
            indices = np.asarray(solver.side_joint_indices[side], dtype=np.intp)
            resolved = solver.side_joint_speed_limits_rad_s[side]
            assert np.all(resolved > 0.0)
            assert np.all(resolved <= urdf[indices])
            assert np.all(resolved <= 1.0)
