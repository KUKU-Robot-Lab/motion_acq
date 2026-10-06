"""RH56F1 grip guard: stop the glove from driving a blocked finger further into an object.

10.06 cup grasp (left hand, bag_left_1006_1010): the glove asked for a full fist, the fingers
stopped on the cup and kept pushing towards the target. force_actual reached 1346-1715 g
against a force_set of 600 g, the six actuators drew 6.5 A together (manual: max grip
4.0 A @ 24 V), and 0.15 s after contact the hand stopped answering (readings frozen, locked
closed until a power cycle). The haptic brakes switched on in the same 0.1 s: too late for an
operator to react, and a squeezing hand pushes through them.

So the robot side protects itself (user, 10.06): per joint, once its motor force or current
says it is pressing on something, the command may close no further than the measured
position plus a small margin (it keeps the grip), and above a hard level it backs off a
little. Opening is never limited, so the finger follows the operator as soon as they let go.
All joints close towards larger rad (fingers curl, thumb_2 bends, thumb_1 swings across the
palm), so the guard is an upper bound per joint.

Two additions from the literature (rl_ws/repo/teleop_refs, 10.06):
* Contact speed (Tan, Xie, Correll, arXiv 2603.08988, RH56DFX): the hand does not slow down
  before contact, so the force overshoot past its force threshold grows with the commanded speed
  (up to +1618 % at full speed, small at 1/40 of it); fast in free space, slow in contact removes
  it. A joint whose tip, force or current shows contact gets contact_speed until it lets go.
* Force bias (GR00T-WholeBodyControl Inspire bridge; our index reads ~96 g at rest): force
  thresholds are relative to the joint's reading at rest, learnt while the hand is home and open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

JOINTS = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
TIP_JOINTS = {"thumb": ("thumb_1", "thumb_2"), "index": ("index_1",), "middle": ("middle_1",),
              "ring": ("ring_1",), "pinky": ("pinky_1",)}


@dataclass(frozen=True)
class GripGuardConfig:
    enabled: bool = True
    force_on_g: float = 400.0  # force_actual (g): pressing on something
    force_off_g: float = 250.0  # ... released (hysteresis)
    force_hard_g: float = 800.0  # back off above this
    current_on_ma: float = 800.0  # current_actual (mA) per actuator, same roles
    current_off_ma: float = 500.0
    current_hard_ma: float = 1100.0
    total_current_ma: float = 3500.0  # all six together (manual max grip 4.0 A): back off the loaded joints
    margin_rad: float = 0.03  # held: command at most measured + margin (keeps the grip)
    backoff_rad: float = 0.03  # hard: command at most measured - backoff
    stale_s: float = 0.3
    # contact speed (SetSpeed1 units, 2000 = full stroke in 1 s): from the first sign of contact
    contact_speed: float = 300.0
    contact_force_g: float = 120.0  # above the rest reading
    contact_current_ma: float = 450.0  # 10.06 free motion <= 320 mA
    contact_tip_n: float = 0.2
    bias_samples: float = 50.0  # rest readings needed before the bias is used

    def __post_init__(self) -> None:
        if not 0 < self.force_off_g < self.force_on_g < self.force_hard_g:
            raise ValueError("grip_guard: need 0 < force_off_g < force_on_g < force_hard_g")
        if not 0 < self.current_off_ma < self.current_on_ma < self.current_hard_ma:
            raise ValueError("grip_guard: need 0 < current_off_ma < current_on_ma < current_hard_ma")
        if self.total_current_ma <= 0 or self.margin_rad < 0 or self.backoff_rad < 0 or self.stale_s <= 0:
            raise ValueError("grip_guard: total_current_ma, stale_s > 0 and margin_rad, backoff_rad >= 0")
        if not 0 < self.contact_speed <= 4000 or min(self.contact_force_g, self.contact_current_ma, self.contact_tip_n) <= 0:
            raise ValueError("grip_guard: contact_speed in (0, 4000], contact thresholds > 0")
        if self.contact_force_g >= self.force_on_g or self.contact_current_ma >= self.current_on_ma:
            raise ValueError("grip_guard: contact (slow down) must come before the hold levels")


def grip_guard_config(raw: Mapping | None) -> GripGuardConfig:
    raw = dict(raw or {})
    known = set(GripGuardConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"grip_guard: unknown keys {unknown}")
    enabled = bool(raw.pop("enabled", True))
    return GripGuardConfig(enabled=enabled, **{k: float(v) for k, v in raw.items()})


def joint_values(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """GetForceAct1 / GetCurrentAct1: joint_names like l_hj_index_1 (or bare index_1)."""
    return {str(n).split("hj_")[-1]: float(v) for n, v in zip(names, values)}


@dataclass
class GripGuard:
    config: GripGuardConfig = field(default_factory=GripGuardConfig)
    force: tuple[dict[str, float], float] | None = None
    current: tuple[dict[str, float], float] | None = None
    engaged: dict[str, str] = field(default_factory=dict)  # joint -> "hold" | "hard"
    tips: tuple[dict[str, float], float] | None = None  # fingertip normal force (N)
    in_contact: set[str] = field(default_factory=set)  # joints at contact speed
    _rest: dict[str, list[float]] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)

    def on_force(self, names: Sequence[str], values: Sequence[float], t: float) -> None:
        self.force = (joint_values(names, values), t)

    def on_tips(self, tips_n: Mapping[str, float], t: float) -> None:
        self.tips = (dict(tips_n), t)

    def learn_rest(self, t: float) -> None:
        """Called while the hand is home, open and still: the force reading of a free joint."""
        force = self._fresh(self.force, t)
        for joint, value in force.items():
            samples = self._rest.setdefault(joint, [])
            samples.append(value)
            del samples[:-500]
            if len(samples) >= self.config.bias_samples:
                self.bias[joint] = sorted(samples)[len(samples) // 2]

    def relative_force(self, t: float) -> dict[str, float]:
        return {j: v - self.bias.get(j, 0.0) for j, v in self._fresh(self.force, t).items()}

    def on_current(self, names: Sequence[str], values: Sequence[float], t: float) -> None:
        self.current = (joint_values(names, values), t)

    def reset(self) -> None:
        self.engaged = {}
        self.in_contact = set()

    def tips_now(self, t: float) -> dict[str, float]:
        return self._fresh(self.tips, t)

    def current_now(self, t: float) -> dict[str, float] | None:
        """Actuator currents (mA) if fresh (record / analysis)."""
        return self._fresh(self.current, t) or None

    def _fresh(self, item, t: float) -> dict[str, float]:
        return item[0] if item is not None and t - item[1] <= self.config.stale_s else {}

    def _level(self, joint: str, force: Mapping[str, float], current: Mapping[str, float], overload: bool) -> str | None:
        c = self.config
        f, i = force.get(joint, 0.0), abs(current.get(joint, 0.0))
        if f >= c.force_hard_g or i >= c.current_hard_ma or (overload and (f >= c.force_off_g or i >= c.current_off_ma)):
            return "hard"
        if f >= c.force_on_g or i >= c.current_on_ma:
            return "hold"
        if joint in self.engaged and (f >= c.force_off_g or i >= c.current_off_ma):
            return "hold"  # still pressing: hysteresis keeps the hold until both readings drop
        return None

    def ceilings(self, measured_rad: Mapping[str, float] | None, t: float,
                 skip: Sequence[str] = ()) -> dict[str, float]:
        """{joint: largest command allowed now}; empty when nothing presses (or no fresh readings).
        skip: joints held by the firmware force loop (grip_mode), not position-commanded."""
        c = self.config
        force, current = self.relative_force(t), self._fresh(self.current, t)
        if not c.enabled or not measured_rad or not (force or current):
            self.engaged = {}
            return {}
        overload = sum(abs(v) for v in current.values()) >= c.total_current_ma
        out, engaged = {}, {}
        for joint in JOINTS:
            if joint not in measured_rad or joint in skip:
                continue
            level = self._level(joint, force, current, overload)
            if level is None:
                continue
            engaged[joint] = level
            q = float(measured_rad[joint])
            out[joint] = q - c.backoff_rad if level == "hard" else q + c.margin_rad
        self.engaged = engaged
        return out

    def speeds(self, t: float, free_speed: float) -> dict[str, float]:
        """{joint: SetSpeed1 value}: contact_speed from the first sign of contact (tip, force or
        current) until all three are back below half their contact levels; free_speed otherwise."""
        c = self.config
        if not c.enabled:
            self.in_contact = set()
            return {j: free_speed for j in JOINTS}
        force, current = self.relative_force(t), self._fresh(self.current, t)
        tips = self._fresh(self.tips, t)
        tip_of = {j: tips.get(f, 0.0) for f, joints in TIP_JOINTS.items() for j in joints}
        contact = set()
        for j in JOINTS:
            f, i, tip = force.get(j, 0.0), abs(current.get(j, 0.0)), tip_of.get(j, 0.0)
            if j in self.engaged or f >= c.contact_force_g or i >= c.contact_current_ma or tip >= c.contact_tip_n:
                contact.add(j)
            elif j in self.in_contact and (f >= c.contact_force_g / 2 or i >= c.contact_current_ma / 2
                                           or tip >= c.contact_tip_n / 2):
                contact.add(j)
        self.in_contact = contact
        return {j: (c.contact_speed if j in contact else free_speed) for j in JOINTS}
