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
from motion_acq.hand.feedback import FeedbackConfig, feedback_config
from motion_acq.hand.rh56f1 import Rh56f1Map

DEFAULT_RETARGET = Path(__file__).resolve().parents[3] / "configs" / "hands" / "nova2_to_rh56f1.yaml"


@dataclass(frozen=True)
class Example:
    prompt: str
    target: Mapping[str, float]  # robot pose (rad) for this operator pose


@dataclass(frozen=True)
class HandRetargetConfig:
    inputs: tuple[str, ...]
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

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(self.limits_rad)

    @property
    def closed_rad(self) -> dict[str, float]:
        """Flexion joints (home at their lower end): where they cannot curl further."""
        return {j: hi for j, (lo, hi) in self.limits_rad.items() if abs(self.home_rad[j] - lo) < 1e-9}


def load_hand_retarget_config(path: Path = DEFAULT_RETARGET) -> HandRetargetConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    inputs = tuple(str(n) for n in raw["inputs"])
    limits = {str(j): (float(v[0]), float(v[1])) for j, v in raw["limits_rad"].items()}
    for j, (lo, hi) in limits.items():
        if not lo < hi:
            raise ValueError(f"limits_rad {j}: {lo} must be below {hi}")
    home = {str(k): float(v) for k, v in raw["home_rad"].items()}
    if set(home) != set(limits):
        raise ValueError("home_rad and limits_rad must list the same joints")
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
    return HandRetargetConfig(
        inputs=inputs, examples=examples, limits_rad=limits, home_rad=home,
        min_cutoff_hz=float(flt.get("min_cutoff_hz", 20.0)), beta=float(flt.get("beta", 0.0)),
        d_cutoff_hz=float(flt.get("d_cutoff_hz", 1.0)),
        max_velocity_rad_s=float(lim.get("max_velocity_rad_s", 8.0)),
        max_acceleration_rad_s2=float(lim.get("max_acceleration_rad_s2", 400.0)),
        max_step_dt_s=float(lim.get("max_step_dt_s", 0.1)),
        rate_hz=float(raw.get("rate_hz", 120.0)), stale_s=float(raw.get("stale_s", 0.2)),
        driver_speed=int(drv.get("speed", 2000)), driver_force=int(drv.get("force", 600)),
        feedback=feedback_config(raw.get("feedback")),
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
        if set(calibration.model.joints) != set(config.joints):
            raise ValueError(f"calibration maps {sorted(calibration.model.joints)}, the hand has "
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
        self._last_t: float | None = None
        self._filters = {j: OneEuroFilter(config.min_cutoff_hz, config.beta, config.d_cutoff_hz)
                         for j in config.joints}
        self._limiters = {j: RateLimiter(config.max_velocity_rad_s, config.max_acceleration_rad_s2)
                          for j in config.joints}
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

    def target(self, signals: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """(raw map output, limited and amplitude-scaled target) for one glove sample."""
        raw = self.calibration.predict(signals)
        _, self.missing_inputs = self.calibration.model.vector(signals)
        home = self.config.home_rad
        out = {}
        for j, (lo, hi) in self.config.limits_rad.items():
            q = min(max(raw[j], lo), hi)
            out[j] = home[j] + self.amplitude * (q - home[j])
        return raw, out

    def step(self, signals: Mapping[str, float] | None, t_s: float) -> HandStep:
        if self.state is HandState.IDLE:
            return HandStep(HandState.IDLE, None, None, None, None, None)
        dt = 0.0 if self._last_t is None else min(max(t_s - self._last_t, 0.0), self.config.max_step_dt_s)
        self._last_t = t_s
        if signals is None:
            self.state = HandState.HOLD
            for lim in self._limiters.values():
                lim.stop()
            return HandStep(HandState.HOLD, None, None, None, self.command(), self._last_registers)
        self.state = HandState.RUNNING
        raw, q_target = self.target(signals)
        q_cmd = {}
        for j in self.config.joints:
            smoothed = self._filters[j](q_target[j], t_s)
            q_cmd[j] = self._limiters[j](smoothed, dt)
        self._last_registers = self.hand_map.to_registers(q_cmd, side=self.side)
        inputs = {n: float(signals[n]) for n in self.calibration.model.inputs if n in signals}
        return HandStep(HandState.RUNNING, inputs, raw, q_target, q_cmd, self._last_registers)
