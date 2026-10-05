"""RH56F1 sensors -> Nova 2 haptics (STEP 3), free of ROS.

The RH56F1 moves 6 joints and reports per-finger tip forces (TouchData1,
0.01 N per count, pinky first), palm forces and a force per joint
(GetForceAct1, joint order from its joint_names). The Nova 2 reads far more
joints than the hand can follow. Its feedback narrows the operator to what the
hand can do and lets him feel contact (10.05 user):

* brake (thumb, index, middle, ring; flexion only, no pinky brake): a robot
  finger that touches something (tip force) or is held back (joint force while
  its command is ahead of its position) brakes the matching glove finger in
  proportion; a robot finger at its closed end brakes it too (the operator
  cannot curl further than the robot). Pinky folds into the ring brake.
* squeeze (palm strap): a constant hold (strap_hold) so the glove sits firmly and the
  same way as when it was calibrated, plus palm contact on top.
* vibration (thumb tip, index tip, palm index side, palm pinky side): a short
  pulse when a finger first touches.

Levels are 0..1 here; the glove driver takes percent (senseglove_ros
convention, maxPosition 100) in the order of HAPTIC_JOINTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

TIP_FINGERS = ("pinky", "ring", "middle", "index", "thumb")  # TouchData1 finger_forces order
TOUCH_N_PER_COUNT = 0.01  # 1024 = 10.24 N (RH56F1 manual 2.5.22, sim2real rh56f1_sensor_watch)
NO_CONTACT = 65535  # sentinel of an unread tactile cell
BRAKES = ("thumb", "index", "middle", "ring")
VIBRATIONS = ("thumb_dip", "index_dip", "palm_index", "palm_pinky")
# forward_command_controller joint order of motion_acq_hand config/nova2_<side>_controllers.yaml
HAPTIC_JOINTS = ("thumb_brake", "index_brake", "middle_brake", "ring_brake", "palm_strap") + VIBRATIONS
# glove brake <- robot fingers (tip force) and robot joints (held back / at its closed end)
BRAKE_SOURCES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "thumb": (("thumb",), ("thumb_2",)),
    "index": (("index",), ("index_1",)),
    "middle": (("middle",), ("middle_1",)),
    "ring": (("ring", "pinky"), ("ring_1", "pinky_1")),
}
VIBRATION_OF = {"thumb": "thumb_dip", "index": "index_dip", "middle": "palm_index",
                "ring": "palm_pinky", "pinky": "palm_pinky"}


@dataclass(frozen=True)
class FeedbackConfig:
    enabled: bool = True
    tip_on_n: float = 0.3  # tip force where the brake starts
    tip_full_n: float = 3.0  # tip force for the full brake
    joint_force_on: float = 150.0  # joint force (driver units, g) where "held back" starts;
    joint_force_full: float = 450.0  # free motion stays under ~70 (sim2real RH56F1_TUNING_0930)
    press_rad: float = 0.05  # command ahead of position (closing) for the joint force to count
    limit_rad: float = 0.03  # command within this of the closed end -> limit brake
    limit_level: float = 1.0  # brake at the closed end (0 = off)
    brake_on: float = 0.15  # a brake below this stays off; once on it holds down to brake_off
    brake_off: float = 0.05
    max_brake: float = 1.0
    palm_on_n: float = 0.5
    palm_full_n: float = 5.0
    max_squeeze: float = 0.5  # the strap squeezes the wrist: half force at most
    pulse_s: float = 0.08
    pulse_level: float = 0.6
    sensor_stale_s: float = 0.3  # older sensor data counts as no contact
    strap_hold: float = 0.0  # strap squeeze while following (holds the glove on the palm, 10.05 user)

    def __post_init__(self) -> None:
        pairs = ((self.tip_on_n, self.tip_full_n), (self.joint_force_on, self.joint_force_full),
                 (self.palm_on_n, self.palm_full_n))
        if any(not 0.0 <= lo < hi for lo, hi in pairs):
            raise ValueError("feedback thresholds need 0 <= on < full")
        if not 0.0 <= self.brake_off <= self.brake_on <= 1.0:
            raise ValueError("feedback needs 0 <= brake_off <= brake_on <= 1")
        if self.strap_hold > self.max_squeeze:
            raise ValueError("feedback strap_hold must not exceed max_squeeze")
        for name in ("max_brake", "max_squeeze", "pulse_level", "limit_level", "strap_hold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"feedback {name} must be in [0, 1]")


def feedback_config(raw: Mapping | None) -> FeedbackConfig:
    """FeedbackConfig from the retarget YAML 'feedback' block (missing keys keep their defaults)."""
    raw = dict(raw or {})
    known = set(FeedbackConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown feedback keys {unknown}")
    values: dict[str, Any] = {k: (bool(v) if k == "enabled" else float(v)) for k, v in raw.items()}
    return FeedbackConfig(**values)


def ramp(x: float, on: float, full: float) -> float:
    return min(max((x - on) / (full - on), 0.0), 1.0)


def tip_forces_n(finger_forces: Sequence[int]) -> dict[str, float]:
    """TouchData1 finger_forces (pinky first, counts) -> {finger: N}; unread cells are 0."""
    if len(finger_forces) != len(TIP_FINGERS):
        raise ValueError(f"{len(finger_forces)} finger forces, expected {len(TIP_FINGERS)}")
    return {f: 0.0 if int(v) in (NO_CONTACT, -1) else max(int(v), 0) * TOUCH_N_PER_COUNT
            for f, v in zip(TIP_FINGERS, finger_forces)}


def palm_force_n(palm_data: Sequence[int]) -> float:
    """Largest palm normal force (EtherCAT: 3 areas x (normal, tangential, direction)), N."""
    normals = [int(v) for i, v in enumerate(palm_data) if i % 3 == 0 and int(v) not in (NO_CONTACT, -1)]
    return max([max(v, 0) * TOUCH_N_PER_COUNT for v in normals], default=0.0)


@dataclass(frozen=True)
class Haptics:
    brake: Mapping[str, float]
    squeeze: float
    vibration: Mapping[str, float]

    def efforts(self) -> list[float]:
        """Percent per HAPTIC_JOINTS entry (the glove driver's effort command)."""
        levels = [self.brake[b] for b in BRAKES] + [self.squeeze] + [self.vibration[v] for v in VIBRATIONS]
        return [round(100.0 * lv, 1) for lv in levels]

    def to_dict(self) -> dict:
        return {"brake": dict(self.brake), "squeeze": self.squeeze, "vibration": dict(self.vibration)}


HAPTICS_MIN_PERIOD_S = 1.0 / 60.0  # the glove driver's update rate
HAPTICS_BEAT_S = 0.25  # re-send period of an unchanged command; the patched glove driver releases after 1 s


def heartbeat(efforts: list[float], beat: bool) -> list[float]:
    """Efforts with the on levels 0.01 % lower on every other send (visible as a change)."""
    return [float(e) - 0.01 if beat and e > 0.0 else float(e) for e in efforts]


def haptics_topic_for(glove_topic: str) -> str:
    """<glove ns>/senseglove_states -> <glove ns>/haptics_controller/commands."""
    return f"{glove_topic.rsplit('/', 1)[0]}/haptics_controller/commands"


def strap_only(level: float) -> list[float]:
    """Efforts (percent) with only the wrist strap on: holds the glove during calibration."""
    return [100.0 * level if j == "palm_strap" else 0.0 for j in HAPTIC_JOINTS]


OFF = Haptics({b: 0.0 for b in BRAKES}, 0.0, {v: 0.0 for v in VIBRATIONS})


@dataclass
class HapticFeedback:
    config: FeedbackConfig
    closed_rad: Mapping[str, float]  # each robot joint's closed end (retarget joints closed_rad)
    tips: tuple[dict[str, float], float] | None = None  # (N per finger, t)
    palm: tuple[float, float] | None = None
    joint_force: tuple[dict[str, float], float] | None = None
    _brake: dict[str, float] = field(default_factory=lambda: {b: 0.0 for b in BRAKES})
    _touching: dict[str, bool] = field(default_factory=lambda: {f: False for f in TIP_FINGERS})
    _pulse_until: dict[str, float] = field(default_factory=lambda: {v: -math.inf for v in VIBRATIONS})

    # -- sensors ----------------------------------------------------------
    def on_touch(self, finger_forces: Sequence[int], palm_data: Sequence[int], t: float) -> None:
        self.tips = (tip_forces_n(finger_forces), t)
        self.palm = (palm_force_n(palm_data), t)

    def on_joint_force(self, names: Sequence[str], values: Sequence[int], t: float) -> None:
        """GetForceAct1: joint_names like r_hj_index_1 (or bare index_1), driver units."""
        self.joint_force = ({str(n).split("hj_")[-1]: float(v) for n, v in zip(names, values)}, t)

    def _fresh(self, item, t: float):
        return item[0] if item is not None and t - item[1] <= self.config.sensor_stale_s else None

    def sensors(self, t: float) -> dict:
        """What the hand reports now (record / dataset), None where stale."""
        tips = self._fresh(self.tips, t)
        force = self._fresh(self.joint_force, t)
        return {"tip_force_n": None if tips is None else {f: round(v, 3) for f, v in tips.items()},
                "palm_force_n": self._fresh(self.palm, t),
                "joint_force": force}

    # -- output -----------------------------------------------------------
    def reset(self) -> None:
        self._brake = {b: 0.0 for b in BRAKES}
        self._touching = {f: False for f in TIP_FINGERS}
        self._pulse_until = {v: -math.inf for v in VIBRATIONS}

    def update(self, t: float, *, active: bool, q_command: Mapping[str, float] | None,
               q_measured: Mapping[str, float] | None) -> Haptics:
        """Levels for this tick. Not active (hand not following the glove) -> all off."""
        cfg = self.config
        if not cfg.enabled or not active:
            self.reset()
            return OFF
        tips = self._fresh(self.tips, t) or {}
        force = self._fresh(self.joint_force, t) or {}
        brake = {}
        for b, (fingers, joints) in BRAKE_SOURCES.items():
            level = max((ramp(tips.get(f, 0.0), cfg.tip_on_n, cfg.tip_full_n) for f in fingers), default=0.0)
            for j in joints:
                level = max(level, self._joint_level(j, q_command, q_measured, force))
            level = min(level, cfg.max_brake)
            held = self._brake[b] > 0.0 and level >= cfg.brake_off
            brake[b] = level if level >= cfg.brake_on or held else 0.0
        self._brake = dict(brake)
        vibration = {v: 0.0 for v in VIBRATIONS}
        for f in TIP_FINGERS:
            touching = tips.get(f, 0.0) >= cfg.tip_on_n
            if touching and not self._touching[f]:
                self._pulse_until[VIBRATION_OF[f]] = t + cfg.pulse_s
            self._touching[f] = touching
        for v, until in self._pulse_until.items():
            if t < until:
                vibration[v] = cfg.pulse_level
        palm = self._fresh(self.palm, t) or 0.0
        contact = (cfg.max_squeeze - cfg.strap_hold) * ramp(palm, cfg.palm_on_n, cfg.palm_full_n)
        squeeze = cfg.strap_hold + contact
        return Haptics(brake, round(squeeze, 3), vibration)

    def _joint_level(self, joint: str, q_command, q_measured, force: Mapping[str, float]) -> float:
        cfg = self.config
        if q_command is None or joint not in q_command:
            return 0.0
        level = 0.0
        closed = self.closed_rad.get(joint)
        if cfg.limit_level > 0.0 and closed is not None and q_command[joint] >= closed - cfg.limit_rad:
            level = cfg.limit_level
        if q_measured is not None and joint in q_measured and joint in force:
            ahead = q_command[joint] - q_measured[joint]
            if ahead >= cfg.press_rad:
                level = max(level, ramp(abs(force[joint]), cfg.joint_force_on, cfg.joint_force_full))
        return level
