"""Position + force closed loop per finger: free = firmware mode 0 (position), holding = mode 1 (force).

10.06 measurements on both hands (docs/RH56F1_HAND_TUNING.md):
* mode 0 follows the angle exactly but in contact ignores force_set and pushes to 1.1-1.85 kg; how much
  force a position offset gives depends on the object and the link that touches (3-8x apart), so no
  position margin sets a grip force;
* mode 1 holds force_set within a few % on every finger (thumb_2 reads 0.64 x force_set) with
  0-250 mA, but ignores the angle (closes to the end in free space, does not open on an angle).

So a finger runs in mode 0 until it touches and the operator closes further than the robot finger
is (the object stops it); then that finger alone switches to mode 1 with a grip force from how far
the operator closes past it (DEX-Mouse-like penetration), and back to mode 0 once the operator opens
past it again. Contact is read from the actuator force sensor and current (link 1 usually touches
before the fingertip, where the tactile sensor is); while only the actuator sees force and the tip
sensor does not, the contact is on link 1 whose shorter lever turns the same actuator force into a
larger contact force (manual 2.5.12), so the set force is scaled down. Joints not in mode 0 get
their measured angle as target, so the switch back to mode 0 does not move them.

The firmware mode is changed over SDO by the hand driver (/hand_<s>/finger_mode_set) and confirmed by
/hand_<s>/finger_mode; a switch not confirmed in ack_timeout_s is undone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

MODE_POSITION, MODE_FORCE = 0, 1
TIP_OF = {"index_1": "index", "middle_1": "middle", "ring_1": "ring", "pinky_1": "pinky", "thumb_2": "thumb"}


class GripState(str, Enum):
    FREE = "free"  # mode 0, following the glove
    ENTERING = "entering"  # mode 1 requested, not confirmed yet
    HOLD = "hold"  # mode 1, holding force_set
    LEAVING = "leaving"  # mode 0 requested, not confirmed yet


@dataclass(frozen=True)
class GripModeConfig:
    enabled: bool = True
    joints: tuple[str, ...] = ("index_1", "middle_1", "ring_1", "pinky_1", "thumb_2")  # thumb_1: not measured yet
    enter_force_g: float = 150.0  # actuator force over rest: touching
    enter_current_ma: float = 450.0
    enter_push_rad: float = 0.05  # ... and the operator closes this much past the robot finger
    exit_open_rad: float = 0.08  # operator opens this much short of it: back to position
    min_hold_s: float = 0.2  # no flapping between the modes
    base_g: float = 300.0  # grip force at enter_push_rad
    gain_g_per_rad: float = 1500.0  # more grip the further the operator closes past the finger
    min_g: float = 150.0
    max_g: float = 800.0
    thumb_2_scale: float = 1.55  # mode 1 thumb_2 holds ~0.64 x force_set (10.06, both hands)
    proximal_scale: float = 0.7  # actuator loaded but tip sensor quiet: link 1 contact, shorter lever
    tip_contact_n: float = 0.2
    ack_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.min_g <= self.base_g <= self.max_g <= 1000:
            raise ValueError("grip_mode: need 0 < min_g <= base_g <= max_g <= 1000")
        if self.enter_push_rad <= 0 or self.exit_open_rad <= 0 or self.ack_timeout_s <= 0:
            raise ValueError("grip_mode: enter_push_rad, exit_open_rad, ack_timeout_s > 0")
        if not 0 < self.proximal_scale <= 1 or self.thumb_2_scale <= 0:
            raise ValueError("grip_mode: proximal_scale in (0, 1], thumb_2_scale > 0")


def grip_mode_config(raw: Mapping | None) -> GripModeConfig:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(GripModeConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"grip_mode: unknown keys {unknown}")
    enabled = bool(raw.pop("enabled", True))
    joints = tuple(str(j) for j in raw.pop("joints", GripModeConfig.joints))
    bad = [j for j in joints if j not in TIP_OF]
    if bad:
        raise ValueError(f"grip_mode: joints {bad} cannot hold by force (allowed: {sorted(TIP_OF)})")
    return GripModeConfig(enabled=enabled, joints=joints, **{k: float(v) for k, v in raw.items()})


@dataclass
class GripStep:
    modes: dict[str, int] | None  # joint -> firmware mode to request now (None: nothing to send)
    force_g: dict[str, float]  # joint -> force_set for the held / entering joints (firmware units)
    pinned: set[str]  # joints whose angle target must be their measured angle (not in mode 0 following)
    released: set[str] = field(default_factory=set)  # joints back in mode 0 this tick: restart their command there


@dataclass
class GripMode:
    config: GripModeConfig = field(default_factory=GripModeConfig)
    state: dict[str, GripState] = field(default_factory=dict)
    since: dict[str, float] = field(default_factory=dict)
    actual: dict[str, int] = field(default_factory=dict)  # joint -> firmware mode read back
    force_g: dict[str, float] = field(default_factory=dict)
    failures: int = 0

    def on_modes(self, modes: Mapping[str, int]) -> None:
        self.actual = {j: int(m) for j, m in modes.items()}

    def active(self) -> bool:
        return any(s is not GripState.FREE for s in self.state.values())

    def _set(self, joint: str, state: GripState, t: float) -> None:
        self.state[joint], self.since[joint] = state, t

    def _grip(self, joint: str, push: float, tip_n: float) -> float:
        c = self.config
        f = min(max(c.base_g + c.gain_g_per_rad * (push - c.enter_push_rad), c.min_g), c.max_g)
        if tip_n < c.tip_contact_n:
            f *= c.proximal_scale
        if joint == "thumb_2":
            f *= c.thumb_2_scale
        return min(f, 1000.0)

    def release_all(self, t: float) -> GripStep:
        """Leaving follow (home, disable, fault): every held joint back to position, pinned until confirmed."""
        modes, pinned, released = {}, set(), set()
        for j, s in list(self.state.items()):
            if s in (GripState.ENTERING, GripState.HOLD):
                modes[j] = MODE_POSITION
                self._set(j, GripState.LEAVING, t)
            if self.state[j] is GripState.LEAVING:
                if self.actual.get(j) == MODE_POSITION:
                    self._set(j, GripState.FREE, t)
                    released.add(j)
                else:
                    pinned.add(j)
                    if t - self.since[j] > self.config.ack_timeout_s:
                        modes[j] = MODE_POSITION
                        self.since[j] = t
        return GripStep(modes or None, {}, pinned, released)

    def update(self, t: float, q_cmd: Mapping[str, float], q_meas: Mapping[str, float],
               force_rel: Mapping[str, float], current: Mapping[str, float],
               tips_n: Mapping[str, float]) -> GripStep:
        c = self.config
        if not c.enabled:
            return self.release_all(t)
        modes: dict[str, int] = {}
        pinned: set[str] = set()
        released: set[str] = set()
        forces: dict[str, float] = {}
        for j in c.joints:
            if j not in q_cmd or j not in q_meas:
                continue
            s = self.state.get(j, GripState.FREE)
            push = float(q_cmd[j]) - float(q_meas[j])  # > 0: the operator closes past the robot finger
            touching = force_rel.get(j, 0.0) >= c.enter_force_g or abs(current.get(j, 0.0)) >= c.enter_current_ma
            tip = tips_n.get(TIP_OF[j], 0.0)
            if s is GripState.FREE:
                # only with a driver that reports the modes (/hand_<s>/finger_mode): else stay in position
                if self.actual and touching and push >= c.enter_push_rad:
                    modes[j] = MODE_FORCE
                    self._set(j, GripState.ENTERING, t)
                    s = GripState.ENTERING
            elif s is GripState.ENTERING:
                if self.actual.get(j) == MODE_FORCE:
                    self._set(j, GripState.HOLD, t)
                    s = GripState.HOLD
                elif t - self.since[j] > c.ack_timeout_s:  # the hand did not take it: stay in position mode
                    self.failures += 1
                    modes[j] = MODE_POSITION
                    self._set(j, GripState.LEAVING, t)
                    s = GripState.LEAVING
            elif s is GripState.HOLD:
                if push <= -c.exit_open_rad and t - self.since[j] >= c.min_hold_s:
                    modes[j] = MODE_POSITION
                    self._set(j, GripState.LEAVING, t)
                    s = GripState.LEAVING
            if s is GripState.LEAVING:
                if self.actual.get(j) == MODE_POSITION:
                    self._set(j, GripState.FREE, t)
                    s = GripState.FREE
                    released.add(j)
                elif t - self.since[j] > c.ack_timeout_s:
                    modes[j] = MODE_POSITION
                    self.since[j] = t
            if s in (GripState.ENTERING, GripState.HOLD):
                forces[j] = self._grip(j, push, tip)
            if s is not GripState.FREE:
                pinned.add(j)
        self.force_g = forces
        return GripStep(modes or None, forces, pinned, released)

    def keep(self) -> GripStep:
        """Glove stale for a moment (HOLD): keep what is held, as it is (a dropout must not drop the object)."""
        pinned = {j for j, s in self.state.items() if s is not GripState.FREE}
        return GripStep(None, dict(self.force_g), pinned)

    def record(self) -> dict | None:
        held = {j: s.value for j, s in self.state.items() if s is not GripState.FREE}
        if not held:
            return None
        return {"state": held, "force_g": {j: round(f) for j, f in self.force_g.items()}}


def slot_list(values: Mapping[str, float | int], slots: Sequence[str], default: int) -> list[int]:
    """{joint: value} -> driver slot order, default where missing."""
    return [int(round(values[j])) if j in values else int(default) for j in slots]
