"""Meta Quest HMD orientation -> pan/tilt head targets (STEP 2).

Pipeline (STEP2 setup doc): relative orientation -> deadband -> One Euro
filter -> scale/sign -> joint window -> velocity/acceleration limit.

Yaw and pitch come from the HMD forward axis in the gravity-aligned tracking
frame (HandUMI convention: x forward, y left, z up), relative to the forward
axis captured at anchor time. This is the pan/tilt-gimbal reading of
``inv(R_anchor) @ R_now``: roll is ignored and yaw stays about the vertical
axis even when the operator anchors while looking down at the table.

Signs in the tracking frame: yaw > 0 turns left, pitch > 0 looks up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from motion_acq.tracking.transforms import quat_to_matrix

# Above this |pitch| the forward axis is close to vertical: its heading is
# dominated by noise (about 11x at 85 deg), so yaw is held instead of updated.
MAX_YAW_PITCH_DEG = 75.0


def wrap_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def hmd_yaw_pitch_deg(pose7: np.ndarray) -> tuple[float | None, float] | None:
    """Absolute (yaw, pitch) of the HMD forward axis.

    Returns None for an unusable pose; yaw is None when |pitch| exceeds
    MAX_YAW_PITCH_DEG and the heading is not trustworthy.
    """
    pose = np.asarray(pose7, dtype=np.float64).reshape(7)
    quaternion = pose[3:7]
    if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) < 1e-6:
        return None
    forward = quat_to_matrix(quaternion)[:, 0]
    horizontal = math.hypot(forward[0], forward[1])
    pitch = math.degrees(math.atan2(forward[2], horizontal))
    if abs(pitch) > MAX_YAW_PITCH_DEG:
        return None, pitch
    return math.degrees(math.atan2(forward[1], forward[0])), pitch


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) for one scalar signal."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        if min_cutoff_hz <= 0.0 or d_cutoff_hz <= 0.0 or beta < 0.0:
            raise ValueError("One Euro needs positive cutoffs and beta >= 0.")
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff_hz = float(d_cutoff_hz)
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, value: float | None = None, t_s: float | None = None) -> None:
        self._x = None if value is None else float(value)
        self._dx = 0.0
        self._t = t_s

    def __call__(self, value: float, t_s: float) -> float:
        value = float(value)
        if self._x is None or self._t is None:
            self._x, self._t = value, t_s
            return value
        dt = t_s - self._t
        if dt <= 0.0:
            return self._x
        dx = (value - self._x) / dt
        self._dx += self._alpha(self.d_cutoff_hz, dt) * (dx - self._dx)
        cutoff = self.min_cutoff_hz + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (value - self._x)
        self._t = t_s
        return self._x


class Deadband:
    """Backlash deadband: the output only moves once the input leaves ±width."""

    def __init__(self, width: float) -> None:
        if width < 0.0:
            raise ValueError("Deadband width must be >= 0.")
        self.width = float(width)
        self._out: float | None = None

    def reset(self, value: float | None = None) -> None:
        self._out = None if value is None else float(value)

    def __call__(self, value: float) -> float:
        value = float(value)
        if self._out is None:
            self._out = value
        elif value > self._out + self.width:
            self._out = value - self.width
        elif value < self._out - self.width:
            self._out = value + self.width
        return self._out


class RateLimiter:
    """Second-order follower: bounded velocity and acceleration, no overshoot."""

    def __init__(self, max_velocity: float, max_acceleration: float) -> None:
        if max_velocity <= 0.0 or max_acceleration <= 0.0:
            raise ValueError("Rate limits must be positive.")
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.position = 0.0
        self.velocity = 0.0

    def reset(self, position: float) -> None:
        self.position = float(position)
        self.velocity = 0.0

    def stop(self) -> None:
        self.velocity = 0.0

    def __call__(self, target: float, dt: float) -> float:
        if dt <= 0.0:
            return self.position
        error = float(target) - self.position
        # Fastest speed that can still brake to zero at the target when the
        # deceleration is applied in steps of dt (discrete braking curve).
        half = 0.5 * self.max_acceleration * dt
        braking = math.sqrt(half * half + 2.0 * self.max_acceleration * abs(error)) - half
        desired = math.copysign(min(self.max_velocity, braking), error)
        step = self.max_acceleration * dt
        self.velocity += float(np.clip(desired - self.velocity, -step, step))
        move = self.velocity * dt
        if abs(move) >= abs(error) and error * move >= 0.0:
            self.position = float(target)
            self.velocity = 0.0
        else:
            self.position += move
        return self.position


@dataclass(frozen=True)
class AxisConfig:
    """One head joint in motor degrees (sim2real deg_to_tick convention)."""

    home_deg: float
    range_deg: float
    sign: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.range_deg <= 0.0:
            raise ValueError("range_deg must be positive.")
        if self.sign not in (-1.0, 1.0):
            raise ValueError("sign must be +1 or -1.")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive.")

    @property
    def lower_deg(self) -> float:
        return self.home_deg - self.range_deg

    @property
    def upper_deg(self) -> float:
        return self.home_deg + self.range_deg

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.lower_deg, self.upper_deg))


@dataclass(frozen=True)
class RetargetConfig:
    pan: AxisConfig
    tilt: AxisConfig
    deadband_deg: float = 0.5
    min_cutoff_hz: float = 1.0
    beta: float = 0.05
    d_cutoff_hz: float = 1.0
    max_velocity_deg_s: float = 60.0
    max_acceleration_deg_s2: float = 300.0
    # A control cycle longer than this (a stalled loop) still integrates only
    # this much time, so a stall never turns into one large step.
    max_step_dt_s: float = 0.06
    # Resume from a HOLD longer than this by re-anchoring at the current head
    # and HMD pose (Quest reconnect, app restart, recenter while lost).
    reanchor_after_hold_s: float = 0.5
    # A frame-to-frame HMD jump larger than this is a tracking-origin change
    # (Quest recenter), not head motion: hold and re-anchor.
    max_hmd_jump_deg: float = 30.0
    pan_enabled: bool = True
    tilt_enabled: bool = True

    def with_axes(self, axes: str) -> RetargetConfig:
        """Restrict to 'pan', 'tilt' or 'both' (staged hardware tests)."""
        if axes not in ("pan", "tilt", "both"):
            raise ValueError(f"axes must be pan, tilt or both, not {axes!r}.")
        return replace(
            self,
            pan_enabled=axes in ("pan", "both"),
            tilt_enabled=axes in ("tilt", "both"),
        )


class HeadState(str, Enum):
    IDLE = "idle"  # not anchored yet: never command
    RUNNING = "running"
    HOLD = "hold"  # HMD lost, unusable or jumped: keep the last command


@dataclass(frozen=True)
class HeadStep:
    state: HeadState
    rel_yaw_deg: float | None
    rel_pitch_deg: float | None
    filtered_yaw_deg: float | None
    filtered_pitch_deg: float | None
    target_pan_deg: float | None
    target_tilt_deg: float | None
    command_pan_deg: float | None
    command_tilt_deg: float | None
    reanchored: bool = False


class _Axis:
    def __init__(self, axis: AxisConfig, config: RetargetConfig, enabled: bool) -> None:
        self.axis = axis
        self.enabled = enabled
        self.deadband = Deadband(config.deadband_deg)
        self.filter = OneEuroFilter(config.min_cutoff_hz, config.beta, config.d_cutoff_hz)
        self.limiter = RateLimiter(config.max_velocity_deg_s, config.max_acceleration_deg_s2)
        self.base_deg = axis.home_deg
        self.target_deg = axis.home_deg
        self.filtered = 0.0

    def anchor(self, head_deg: float, t_s: float) -> None:
        self.base_deg = self.axis.clamp(head_deg)
        self.target_deg = self.base_deg
        self.filtered = 0.0
        self.deadband.reset(0.0)
        self.filter.reset(0.0, t_s)
        self.limiter.reset(self.base_deg)

    def step(self, relative_deg: float, t_s: float, dt: float) -> float:
        self.filtered = self.filter(self.deadband(relative_deg), t_s)
        if self.enabled:
            raw = self.base_deg + self.axis.sign * self.axis.scale * self.filtered
            self.target_deg = self.axis.clamp(raw)
        else:
            self.target_deg = self.base_deg
        return self.limiter(self.target_deg, dt)


class HeadRetargeter:
    """Anchor-relative HMD yaw/pitch -> limited pan/tilt commands.

    Yaw is unwrapped continuously (no flip at ±180 deg from the anchor) and is
    held while the HMD looks near-vertically. HOLD keeps the last command; a
    long HOLD or an HMD jump re-anchors so the head never snaps to a new origin.
    """

    def __init__(self, config: RetargetConfig) -> None:
        self.config = config
        self.state = HeadState.IDLE
        self._anchor_yaw = 0.0
        self._anchor_pitch = 0.0
        self._yaw_unwrapped = 0.0
        self._last_raw_yaw = 0.0
        self._last_pitch = 0.0
        self._last_t: float | None = None
        self._hold_since: float | None = None
        self._pan = _Axis(config.pan, config, config.pan_enabled)
        self._tilt = _Axis(config.tilt, config, config.tilt_enabled)

    @property
    def command_deg(self) -> tuple[float, float]:
        return self._pan.limiter.position, self._tilt.limiter.position

    def anchor(
        self, hmd_pose7: np.ndarray, head_deg: tuple[float, float], t_s: float
    ) -> bool:
        """Capture R_hmd_anchor and R_head_anchor. False if the HMD pose is unusable."""
        angles = hmd_yaw_pitch_deg(hmd_pose7)
        if angles is None or angles[0] is None:
            return False
        yaw, pitch = angles
        self._anchor_yaw = self._yaw_unwrapped = self._last_raw_yaw = yaw
        self._anchor_pitch = self._last_pitch = pitch
        self._pan.anchor(head_deg[0], t_s)
        self._tilt.anchor(head_deg[1], t_s)
        self._last_t = t_s
        self._hold_since = None
        self.state = HeadState.RUNNING
        return True

    def _hold(self, t_s: float) -> HeadStep:
        if self._hold_since is None:
            self._hold_since = t_s
        self.state = HeadState.HOLD
        self._pan.limiter.stop()
        self._tilt.limiter.stop()
        return HeadStep(
            HeadState.HOLD, None, None, None, None,
            self._pan.target_deg, self._tilt.target_deg, *self.command_deg,
        )

    def _reanchor(self, hmd_pose7: np.ndarray, t_s: float) -> HeadStep:
        self.anchor(hmd_pose7, self.command_deg, t_s)
        return HeadStep(
            HeadState.RUNNING, 0.0, 0.0, 0.0, 0.0,
            self._pan.target_deg, self._tilt.target_deg, *self.command_deg,
            reanchored=True,
        )

    def step(self, hmd_pose7: np.ndarray | None, tracked: bool, t_s: float) -> HeadStep:
        if self.state is HeadState.IDLE:
            return HeadStep(HeadState.IDLE, *([None] * 8))
        raw_dt = 0.0 if self._last_t is None else max(t_s - self._last_t, 0.0)
        dt = min(raw_dt, self.config.max_step_dt_s)
        self._last_t = t_s
        angles = hmd_yaw_pitch_deg(hmd_pose7) if tracked and hmd_pose7 is not None else None
        if angles is None:
            return self._hold(t_s)
        assert hmd_pose7 is not None
        raw_yaw, pitch = angles
        long_hold = (
            self._hold_since is not None
            and t_s - self._hold_since > self.config.reanchor_after_hold_s
        )
        jump = abs(pitch - self._last_pitch) > self.config.max_hmd_jump_deg or (
            raw_yaw is not None
            and abs(wrap_deg(raw_yaw - self._last_raw_yaw)) > self.config.max_hmd_jump_deg
        )
        if long_hold or jump:
            if raw_yaw is None:
                return self._hold(t_s)  # cannot anchor while looking vertically
            return self._reanchor(hmd_pose7, t_s)
        self._hold_since = None
        self.state = HeadState.RUNNING
        if raw_yaw is not None:
            self._yaw_unwrapped += wrap_deg(raw_yaw - self._last_raw_yaw)
            self._last_raw_yaw = raw_yaw
        self._last_pitch = pitch
        rel_yaw = self._yaw_unwrapped - self._anchor_yaw
        rel_pitch = pitch - self._anchor_pitch
        pan_cmd = self._pan.step(rel_yaw, t_s, dt)
        tilt_cmd = self._tilt.step(rel_pitch, t_s, dt)
        return HeadStep(
            HeadState.RUNNING,
            rel_yaw, rel_pitch, self._pan.filtered, self._tilt.filtered,
            self._pan.target_deg, self._tilt.target_deg,
            pan_cmd, tilt_cmd,
        )
