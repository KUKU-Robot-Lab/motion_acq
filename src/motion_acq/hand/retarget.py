"""Nova 2 glove angles -> RH56F1 joint targets and registers (STEP 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import yaml

from motion_acq.filters import OneEuroFilter, RateLimiter
from motion_acq.hand.calibration import HandCalibration
from motion_acq.hand.feedback import FeedbackConfig, feedback_config
from motion_acq.hand.nova2 import FeatureSpec, features
from motion_acq.hand.rh56f1 import Rh56f1Map

DEFAULT_RETARGET = Path(__file__).resolve().parents[3] / "configs" / "hands" / "nova2_to_rh56f1.yaml"


@dataclass(frozen=True)
class JointMap:
    joint: str
    feature: str
    open_rad: float
    closed_rad: float

    def target(self, n: float) -> float:
        return self.open_rad + n * (self.closed_rad - self.open_rad)


@dataclass(frozen=True)
class HandRetargetConfig:
    features: tuple[FeatureSpec, ...]
    feature_poses: Mapping[str, tuple[str, str]]
    poses: Mapping[str, str]
    joints: tuple[JointMap, ...]
    home_rad: Mapping[str, float]
    min_span_rad: float = 0.2
    end_margin_open: float = 0.0  # n below this -> 0 (see stretch)
    end_margin_closed: float = 0.0  # n above 1 - this -> 1
    min_cutoff_hz: float = 1.5
    beta: float = 0.3
    d_cutoff_hz: float = 1.0
    max_velocity_rad_s: float = 2.0
    max_acceleration_rad_s2: float = 20.0
    max_step_dt_s: float = 0.1
    rate_hz: float = 30.0
    stale_s: float = 0.2
    driver_speed: int = 2000
    driver_force: int = 600
    feedback: FeedbackConfig = FeedbackConfig()


def load_hand_retarget_config(path: Path = DEFAULT_RETARGET) -> HandRetargetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = tuple(FeatureSpec(n, {k: float(v) for k, v in w.items()}) for n, w in raw["features"].items())
    names = {s.name for s in specs}
    joints = tuple(
        JointMap(j, str(v["feature"]), float(v["open_rad"]), float(v["closed_rad"]))
        for j, v in raw["joints"].items()
    )
    for jm in joints:
        if jm.feature not in names:
            raise ValueError(f"joint {jm.joint} uses unknown feature {jm.feature!r}")
    feature_poses = {k: (str(v[0]), str(v[1])) for k, v in raw["feature_poses"].items()}
    if set(feature_poses) != names:
        raise ValueError("feature_poses must list every feature exactly once")
    poses = {str(k): str(v) for k, v in raw["poses"].items()}
    for open_pose, closed_pose in feature_poses.values():
        if open_pose not in poses or closed_pose not in poses:
            raise ValueError(f"feature_poses names an undefined pose: {open_pose}, {closed_pose}")
    flt, lim, drv = raw.get("filter") or {}, raw.get("limits") or {}, raw.get("driver") or {}
    margins = raw.get("end_margins") or {}
    m_open, m_closed = float(margins.get("open", 0.0)), float(margins.get("closed", 0.0))
    if m_open < 0.0 or m_closed < 0.0 or m_open + m_closed >= 0.9:
        raise ValueError(f"end_margins open {m_open} / closed {m_closed}: each >= 0, together < 0.9")
    home = {k: float(v) for k, v in raw["home_rad"].items()}
    if set(home) != {jm.joint for jm in joints}:
        raise ValueError("home_rad must give every mapped joint")
    return HandRetargetConfig(
        features=specs, feature_poses=feature_poses, poses=poses, joints=joints, home_rad=home,
        min_span_rad=float(raw.get("min_span_rad", 0.2)),
        end_margin_open=m_open, end_margin_closed=m_closed,
        min_cutoff_hz=float(flt.get("min_cutoff_hz", 1.5)), beta=float(flt.get("beta", 0.3)),
        d_cutoff_hz=float(flt.get("d_cutoff_hz", 1.0)),
        max_velocity_rad_s=float(lim.get("max_velocity_rad_s", 2.0)),
        max_acceleration_rad_s2=float(lim.get("max_acceleration_rad_s2", 20.0)),
        max_step_dt_s=float(lim.get("max_step_dt_s", 0.1)),
        rate_hz=float(raw.get("rate_hz", 30.0)), stale_s=float(raw.get("stale_s", 0.2)),
        driver_speed=int(drv.get("speed", 2000)), driver_force=int(drv.get("force", 600)),
        feedback=feedback_config(raw.get("feedback")),
    )


def stretch(n: float, margin_open: float, margin_closed: float) -> float:
    """Calibrated n with both ends reached early: the operator's working fist or open
    hand is a little short of the calibration poses (10.05: 85-89 %)."""
    span = 1.0 - margin_open - margin_closed
    return min(max((n - margin_open) / span, 0.0), 1.0)


class HandState(str, Enum):
    IDLE = "idle"  # not started: never command
    RUNNING = "running"
    HOLD = "hold"  # glove stale/lost: keep the last command
    HOMING = "homing"  # walking to the home pose (start and end of a session)


@dataclass(frozen=True)
class HandStep:
    state: HandState
    features: Mapping[str, float] | None
    normalized: Mapping[str, float] | None
    q_target: Mapping[str, float] | None  # after scaling, before filter/limits
    q_command: Mapping[str, float] | None  # filtered and rate limited (rad)
    registers: list[int] | None  # SetAngle1 slot order


class HandRetargeter:
    """One glove side -> one RH56F1. start() seeds from the measured hand pose."""

    def __init__(
        self,
        config: HandRetargetConfig,
        calibration: HandCalibration,
        hand_map: Rh56f1Map,
        side: str,
        *,
        amplitude: float = 1.0,
    ) -> None:
        if calibration.side != side:
            raise ValueError(f"calibration is for the {calibration.side} hand, not {side}")
        missing = {s.name for s in config.features} - set(calibration.ranges)
        if missing:
            raise ValueError(f"calibration lacks features {sorted(missing)}; recalibrate")
        if not 0.0 < amplitude <= 1.0:
            raise ValueError(f"amplitude must be in (0, 1], not {amplitude}")
        # Fraction of each joint's open->closed travel that is used (first real
        # runs start small); n is scaled before the joint mapping.
        self.amplitude = float(amplitude)
        mapped = {jm.joint for jm in config.joints}
        if mapped != set(hand_map.joint_order):
            raise ValueError(f"mapping covers {sorted(mapped)}, hand has {list(hand_map.joint_order)}")
        self.config = config
        self.calibration = calibration
        self.hand_map = hand_map
        self.side = side
        self.state = HandState.IDLE
        self._last_t: float | None = None
        self._filters = {
            jm.joint: OneEuroFilter(config.min_cutoff_hz, config.beta, config.d_cutoff_hz)
            for jm in config.joints
        }
        self._limiters = {
            jm.joint: RateLimiter(config.max_velocity_rad_s, config.max_acceleration_rad_s2)
            for jm in config.joints
        }
        self._last_registers: list[int] | None = None

    def command(self) -> dict[str, float]:
        return {j: lim.position for j, lim in self._limiters.items()}

    def start(self, measured_rad: Mapping[str, float] | None, t_s: float) -> None:
        """Begin following from the measured hand pose (home if unknown)."""
        start = dict(self.config.home_rad)
        if measured_rad:
            start.update({k: float(v) for k, v in measured_rad.items() if k in start})
        for joint, value in start.items():
            self._limiters[joint].reset(value)
            self._filters[joint].reset(None)
        self._last_t = t_s
        self._last_registers = self.hand_map.to_registers(self.command(), side=self.side)
        self.state = HandState.RUNNING

    def step_to(self, q_target: Mapping[str, float], t_s: float) -> HandStep:
        """Walk the command to a fixed pose (home) under the same rate limits.

        The glove filters restart afterwards, so following begins from here.
        """
        if self.state is HandState.IDLE:
            return HandStep(HandState.IDLE, None, None, None, None, None)
        dt = 0.0 if self._last_t is None else min(max(t_s - self._last_t, 0.0), self.config.max_step_dt_s)
        self._last_t = t_s
        target = {j: float(q_target[j]) for j in self._limiters}
        q_cmd = {j: self._limiters[j](target[j], dt) for j in self._limiters}
        for filt in self._filters.values():
            filt.reset(None)
        self.state = HandState.HOMING
        self._last_registers = self.hand_map.to_registers(q_cmd, side=self.side)
        return HandStep(HandState.HOMING, None, None, target, q_cmd, self._last_registers)

    def step(self, angles: Mapping[str, float] | None, t_s: float) -> HandStep:
        if self.state is HandState.IDLE:
            return HandStep(HandState.IDLE, None, None, None, None, None)
        dt = 0.0 if self._last_t is None else min(max(t_s - self._last_t, 0.0), self.config.max_step_dt_s)
        self._last_t = t_s
        if angles is None:
            self.state = HandState.HOLD
            for lim in self._limiters.values():
                lim.stop()
            return HandStep(HandState.HOLD, None, None, None, self.command(), self._last_registers)
        self.state = HandState.RUNNING
        feats = features(angles, self.config.features)
        cfg = self.config
        norm = {k: stretch(v, cfg.end_margin_open, cfg.end_margin_closed)
                for k, v in self.calibration.normalize(feats).items()}
        q_target, q_cmd = {}, {}
        for jm in self.config.joints:
            q_target[jm.joint] = jm.target(self.amplitude * norm[jm.feature])
            smoothed = self._filters[jm.joint](q_target[jm.joint], t_s)
            q_cmd[jm.joint] = self._limiters[jm.joint](smoothed, dt)
        self._last_registers = self.hand_map.to_registers(q_cmd, side=self.side)
        return HandStep(HandState.RUNNING, feats, norm, q_target, q_cmd, self._last_registers)
