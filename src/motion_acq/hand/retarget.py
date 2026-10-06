"""Nova 2 glove signals -> RH56F1 joint targets and registers (STEP 3).

10.05: the map is learned per operator from example poses (motion_acq.hand.example_map,
calibration v2). Each tick: glove signals (20 angles + thumb-to-finger tip distances)
-> the operator's map -> joint limits -> amplitude around home -> glitch filter and rate
limit -> registers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

import yaml

from motion_acq.filters import OneEuroFilter, RateLimiter
from motion_acq.hand.calibration import HandCalibration
from motion_acq.hand.admittance import AdmittanceConfig, admittance_config
from motion_acq.hand.feedback import FeedbackConfig, feedback_config
from motion_acq.hand.grip_guard import GripGuardConfig, grip_guard_config
from motion_acq.hand.kinematic import KinematicConfig, KinematicRetargeter, TipTables, points_from_signals, table_path
from motion_acq.hand.rh56f1 import Rh56f1Map

DEFAULT_RETARGET = Path(__file__).resolve().parents[3] / "configs" / "hands" / "nova2_to_rh56f1.yaml"


@dataclass(frozen=True)
class Example:
    prompt: str
    target: Mapping[str, float]  # robot pose (rad) for this operator pose


@dataclass(frozen=True)
class HandRetargetConfig:
    groups: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]  # joint group -> (inputs, joints)
    examples: Mapping[str, Example]
    limits_rad: Mapping[str, tuple[float, float]]  # every RH56F1 joint: (min, max) the map may command
    home_rad: Mapping[str, float]
    min_cutoff_hz: float = 20.0
    beta: float = 0.0
    d_cutoff_hz: float = 1.0
    max_velocity_rad_s: float = 8.0
    max_acceleration_rad_s2: float = 400.0
    max_step_dt_s: float = 0.1
    rate_hz: float = 120.0
    stale_s: float = 0.2
    driver_speed: int = 2000
    driver_force: int = 600
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    method: str = "examples"  # "kinematic": fingertip retargeting when the calibration has an alignment
    # reference at [켜기] (10.06 user): once the hand is home, the operator holds this example pose
    # (the robot's home: fingers straight, thumb beside the index) still for reference_s; that take
    # re-zeroes the map and re-aligns the hand model before following. 0 s: off.
    reference_pose: str = "flat"
    reference_s: float = 1.0
    reference_timeout_s: float = 10.0
    kinematic: KinematicConfig = field(default_factory=KinematicConfig)
    grip_guard: GripGuardConfig = field(default_factory=GripGuardConfig)
    admittance: AdmittanceConfig = field(default_factory=AdmittanceConfig)

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(self.limits_rad)

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(n for ins, _ in self.groups.values() for n in ins))

    @property
    def closed_rad(self) -> dict[str, float]:
        """Flexion joints (home at their lower end): where they cannot curl further."""
        return {j: hi for j, (lo, hi) in self.limits_rad.items() if abs(self.home_rad[j] - lo) < 1e-9}


def load_hand_retarget_config(path: Path = DEFAULT_RETARGET) -> HandRetargetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups = {str(g): (tuple(str(n) for n in v["inputs"]), tuple(str(j) for j in v["joints"]))
              for g, v in raw["groups"].items()}
    limits = {str(j): (float(v[0]), float(v[1])) for j, v in raw["limits_rad"].items()}
    for j, (lo, hi) in limits.items():
        if not lo < hi:
            raise ValueError(f"limits_rad {j}: {lo} must be below {hi}")
    home = {str(k): float(v) for k, v in raw["home_rad"].items()}
    if set(home) != set(limits):
        raise ValueError("home_rad and limits_rad must list the same joints")
    grouped = [j for _, joints in groups.values() for j in joints]
    if sorted(grouped) != sorted(limits):
        raise ValueError(f"groups must cover every joint exactly once: {grouped} vs {sorted(limits)}")
    examples = {}
    for pose, entry in raw["examples"].items():
        target = {str(j): float(v) for j, v in entry["target"].items()}
        if set(target) != set(limits):
            raise ValueError(f"example {pose}: target must give every joint {sorted(limits)}")
        out = [j for j, v in target.items() if not limits[j][0] - 1e-9 <= v <= limits[j][1] + 1e-9]
        if out:
            raise ValueError(f"example {pose}: {out} outside limits_rad")
        examples[str(pose)] = Example(str(entry["prompt"]), target)
    if "open" not in examples:
        raise ValueError("examples need an 'open' pose (the quick re-zero pose)")
    flt, lim, drv = raw.get("filter") or {}, raw.get("limits") or {}, raw.get("driver") or {}
    method = str(raw.get("method", "examples"))
    ref = raw.get("reference") or {}
    if str(ref.get("pose", "flat")) not in examples:
        raise ValueError(f"reference pose {ref.get('pose')!r} is not an example pose")
    if method not in ("examples", "kinematic"):
        raise ValueError(f"method must be examples or kinematic, not {method!r}")
    return HandRetargetConfig(
        groups=groups, examples=examples, limits_rad=limits, home_rad=home,
        min_cutoff_hz=float(flt.get("min_cutoff_hz", 20.0)), beta=float(flt.get("beta", 0.0)),
        d_cutoff_hz=float(flt.get("d_cutoff_hz", 1.0)),
        max_velocity_rad_s=float(lim.get("max_velocity_rad_s", 8.0)),
        max_acceleration_rad_s2=float(lim.get("max_acceleration_rad_s2", 400.0)),
        max_step_dt_s=float(lim.get("max_step_dt_s", 0.1)),
        rate_hz=float(raw.get("rate_hz", 120.0)), stale_s=float(raw.get("stale_s", 0.2)),
        driver_speed=int(drv.get("speed", 2000)), driver_force=int(drv.get("force", 600)),
        feedback=feedback_config(raw.get("feedback")),
        method=method,
        reference_pose=str(ref.get("pose", "flat")), reference_s=float(ref.get("seconds", 1.0)),
        reference_timeout_s=float(ref.get("timeout_s", 10.0)),
        kinematic=KinematicConfig(**{k: float(v) for k, v in (raw.get("kinematic") or {}).items()}),
        grip_guard=grip_guard_config(raw.get("grip_guard")),
        admittance=admittance_config(raw.get("admittance")),
    )


class HandState(str, Enum):
    IDLE = "idle"  # not started: never command
    RUNNING = "running"
    HOLD = "hold"  # glove stale/lost: keep the last command
    HOMING = "homing"  # walking to the home pose (start and end of a session)


@dataclass(frozen=True)
class HandStep:
    state: HandState
    features: Mapping[str, float] | None  # the map's glove inputs this tick
    normalized: Mapping[str, float] | None  # the map's raw output (rad), before limits / amplitude
    q_target: Mapping[str, float] | None  # after limits and amplitude, before filter/limits
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
        if set(calibration.joints) != set(config.joints):
            raise ValueError(f"calibration maps {sorted(calibration.joints)}, the hand has "
                             f"{sorted(config.joints)}; recalibrate")
        if not 0.0 < amplitude <= 1.0:
            raise ValueError(f"amplitude must be in (0, 1], not {amplitude}")
        # Fraction of each joint's travel from home that is used (first real runs start small).
        self.amplitude = float(amplitude)
        if set(config.joints) != set(hand_map.joint_order):
            raise ValueError(f"mapping covers {sorted(config.joints)}, hand has {list(hand_map.joint_order)}")
        self.config = config
        self.calibration = calibration
        self.hand_map = hand_map
        self.side = side
        self.state = HandState.IDLE
        self.missing_inputs: list[str] = []
        self.method_used: str | None = None
        self.kinematic = None
        self.set_calibration(calibration)
        self._last_t: float | None = None
        self._filters = {j: OneEuroFilter(config.min_cutoff_hz, config.beta, config.d_cutoff_hz)
                         for j in config.joints}
        self._limiters = {j: RateLimiter(config.max_velocity_rad_s, config.max_acceleration_rad_s2)
                          for j in config.joints}
        self._last_registers: list[int] | None = None

    def set_calibration(self, calibration: HandCalibration) -> None:
        """Use this calibration from now on (the reference take at [켜기] re-zeroes and re-aligns)."""
        if calibration.side != self.side:
            raise ValueError(f"calibration is for the {calibration.side} hand, not {self.side}")
        self.calibration = calibration
        self.kinematic = None
        if self.config.method == "kinematic" and calibration.alignment is not None:
            self.kinematic = KinematicRetargeter(TipTables.load(table_path(self.side)), calibration.alignment,
                                                 self.config.kinematic)

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

    def target(self, signals: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """(raw map output, limited and amplitude-scaled target) for one glove sample."""
        points = points_from_signals(signals) if self.kinematic is not None else None
        if points is not None:
            ratio = self.calibration.thumb_bend_ratio(signals)
            closed = self.config.limits_rad["thumb_2"][1]
            raw = self.kinematic.solve(points, None if ratio is None else ratio * closed)
            self.missing_inputs, self.method_used = [], "kinematic"
        else:  # examples map: the method itself, or no hand model data / alignment this tick
            raw = self.calibration.predict(signals)
            self.missing_inputs, self.method_used = self.calibration.missing_inputs(signals), "examples"
        home = self.config.home_rad
        out = {}
        for j, (lo, hi) in self.config.limits_rad.items():
            q = min(max(raw[j], lo), hi)
            out[j] = home[j] + self.amplitude * (q - home[j])
        return raw, out

    def _cap(self, q_cmd: Mapping[str, float], ceilings: Mapping[str, float] | None) -> dict[str, float]:
        """Grip guard: a pressing joint closes no further than its ceiling (grip_guard.py). The
        limiter restarts from the capped command, so opening follows the glove at once."""
        out = dict(q_cmd)
        for j, top in (ceilings or {}).items():
            if j in out and out[j] > top:
                out[j] = max(float(top), self.config.limits_rad[j][0])
                self._limiters[j].reset(out[j])
        return out

    def step(self, signals: Mapping[str, float] | None, t_s: float,
             ceilings: Mapping[str, float] | None = None, offsets: Mapping[str, float] | None = None) -> HandStep:
        """offsets: rad taken off the operator target per joint (admittance force term)."""
        if self.state is HandState.IDLE:
            return HandStep(HandState.IDLE, None, None, None, None, None)
        dt = 0.0 if self._last_t is None else min(max(t_s - self._last_t, 0.0), self.config.max_step_dt_s)
        self._last_t = t_s
        if signals is None:
            self.state = HandState.HOLD
            for lim in self._limiters.values():
                lim.stop()
            q_hold = self._cap(self.command(), ceilings)
            self._last_registers = self.hand_map.to_registers(q_hold, side=self.side)
            return HandStep(HandState.HOLD, None, None, None, q_hold, self._last_registers)
        self.state = HandState.RUNNING
        raw, q_target = self.target(signals)
        for j, y in (offsets or {}).items():
            if j in q_target:
                q_target[j] = max(q_target[j] - y, self.config.limits_rad[j][0])
        q_cmd = {}
        for j in self.config.joints:
            smoothed = self._filters[j](q_target[j], t_s)
            q_cmd[j] = self._limiters[j](smoothed, dt)
        q_cmd = self._cap(q_cmd, ceilings)
        self._last_registers = self.hand_map.to_registers(q_cmd, side=self.side)
        names = self.calibration.inputs if self.method_used == "examples" else [
            n for n in signals if n.startswith(("knuckle_", "tip_"))]
        inputs = {n: float(signals[n]) for n in names if n in signals}
        return HandStep(HandState.RUNNING, inputs, raw, q_target, q_cmd, self._last_registers)
