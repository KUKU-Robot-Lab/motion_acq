"""Joystick map (10.11 user): each robot joint from one glove angle of that same finger, tuned by eye.

10.11 user: "the thumb only from the Nova's thumb, the index only from the index tip" -- like a joystick.
The thumb has two robot joints:
* thumb_1, the part that swings the thumb across the palm: glove thumb_mcp by default. In both saved
  calibrations it rises open (spread away) -> flat (beside the index) -> thumb_opposed (across the
  palm) (left -0.24 / -0.06 / 0.09, right -0.37 / -0.06 / 0.09 rad), with the same sign on both
  hands, and bending the thumb tip hardly moves it (thumb_bend -0.03 left, -0.13 right). thumb_brake
  does not order those poses on the left glove (0.53 / 0.31 / 0.81), thumb_tip_x moves 10 mm with the
  tip bend alone. The tuning window (motion_acq_hand tune) shows the candidates live and can switch.
* thumb_2, the tip bend: glove thumb_pip.
* index_1 / middle_1 / ring_1: that finger's mcp + pip; pinky_1 follows the ring (no little-finger sensor).

Each joint is a piecewise-linear map through its points (glove reading -> robot angle), clipped at the
ends. The points are named operator poses (POINTS); their robot angles are the example targets of
configs/hands/nova2_to_rh56f1.yaml, their glove readings are captured in the tuning window (default:
the calibration's example-pose medians). With [손가락별 맞춤] sweeps in the calibration, the reading of a
joint whose source is its standard glove inputs has the other fingers' crosstalk taken off (sweeps.py).

The [켜기] reference take (the operator holds the robot's home: fingers straight, thumb beside the
index) shifts every point of a joint by (reading now - reading the map puts at home), when that shift is
within REFERENCE_TOL of the joint's glove span (a shifted glove, e.g. a SenseCom restart; more is a
hand not in the pose).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from motion_acq.hand.sweeps import JOINT_INPUTS as SWEEP_JOINT_INPUTS, SOURCES, FingerSweeps, source_values

JOINTS = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1")  # pinky_1 copies ring_1
FOLLOWS = {"pinky_1": "ring_1"}
JOINT_KO = {"thumb_1": "엄지 회전 (JOINT1, 손바닥 가로지르기)", "thumb_2": "엄지 끝 굽힘 (JOINT2)",
            "index_1": "검지", "middle_1": "중지", "ring_1": "약지·새끼"}
# joint -> (point name, operator pose of the calibration whose median is the default, Korean label)
POINTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "thumb_1": (("spread", "open", "벌림"), ("beside", "flat", "검지 옆"), ("across", "thumb_opposed", "가로지름")),
    "thumb_2": (("straight", "flat", "펴기"), ("bent", "thumb_bend", "굽힘")),
    "index_1": (("straight", "flat", "펴기"), ("bent", "fist", "굽힘")),
    "middle_1": (("straight", "flat", "펴기"), ("bent", "fist", "굽힘")),
    "ring_1": (("straight", "flat", "펴기"), ("bent", "fist", "굽힘")),
}
# joint -> glove sources the window offers (a source is input names joined by "+", summed); first = default
SOURCE_CHOICES: dict[str, tuple[str, ...]] = {
    "thumb_1": ("thumb_mcp", "thumb_brake", "thumb_pip", "thumb_tip_x", "thumb_tip_y", "thumb_tip_z"),
    "thumb_2": ("thumb_pip", "thumb_mcp", "thumb_brake"),
    "index_1": ("index_mcp+index_pip", "index_pip", "index_mcp"),
    "middle_1": ("middle_mcp+middle_pip", "middle_pip", "middle_mcp"),
    "ring_1": ("ring_mcp+ring_pip", "ring_pip", "ring_mcp"),
}
REFERENCE_TOL = 0.3  # fraction of the joint's glove span a reference take may shift it
MIN_SPAN = {"angle": 0.05, "position": 5.0}  # rad / mm between two neighbouring points


class JoystickError(ValueError):
    pass


def source_inputs(source: str) -> tuple[str, ...]:
    return tuple(n.strip() for n in source.split("+") if n.strip())


def _kind(source: str) -> str:
    return "position" if any(n.startswith("thumb_tip_") for n in source_inputs(source)) else "angle"


def raw_reading(source: str, signals: Mapping[str, float]) -> float | None:
    names = source_inputs(source)
    if not names or not all(n in signals for n in names):
        return None
    return float(sum(float(signals[n]) for n in names))


@dataclass(frozen=True)
class JointMap:
    source: str
    glove: tuple[float, ...]  # reading at each point, in POINTS order
    robot: tuple[float, ...]  # robot angle (rad) at each point

    def check(self, joint: str) -> None:
        if len(self.glove) != len(self.robot) or len(self.glove) < 2:
            raise JoystickError(f"{joint}: need the same number (>= 2) of glove and robot points")
        steps = [b - a for a, b in zip(self.glove, self.glove[1:])]
        least = MIN_SPAN[_kind(self.source)]
        if not (all(s >= least for s in steps) or all(s <= -least for s in steps)):
            pts = ", ".join(f"{g:.2f}" for g in self.glove)
            raise JoystickError(f"{JOINT_KO.get(joint, joint)}: 장갑 값이 점 순서대로 한 방향으로 "
                                f"{least:g} 이상씩 변해야 합니다 ({pts})")

    def angle(self, x: float) -> float:
        g, q = self.glove, self.robot
        if g[-1] < g[0]:  # decreasing: walk it reversed
            g, q = g[::-1], q[::-1]
        if x <= g[0]:
            return q[0]
        for (g0, q0), (g1, q1) in zip(zip(g, q), zip(g[1:], q[1:])):
            if x <= g1:
                return q0 + (x - g0) / (g1 - g0) * (q1 - q0)
        return q[-1]

    def glove_at(self, q_target: float) -> float | None:
        """The reading the map puts at robot angle q_target (None: outside its points)."""
        for (g0, q0), (g1, q1) in zip(zip(self.glove, self.robot), zip(self.glove[1:], self.robot[1:])):
            if min(q0, q1) - 1e-9 <= q_target <= max(q0, q1) + 1e-9:
                return g0 if abs(q1 - q0) < 1e-12 else g0 + (q_target - q0) / (q1 - q0) * (g1 - g0)
        return None

    def shifted(self, d: float) -> JointMap:
        return JointMap(self.source, tuple(g + d for g in self.glove), self.robot)

    def to_dict(self) -> dict:
        return {"source": self.source, "glove": [float(g) for g in self.glove],
                "robot": [float(q) for q in self.robot]}

    @classmethod
    def from_dict(cls, raw: Mapping) -> JointMap:
        return cls(str(raw["source"]), tuple(float(g) for g in raw["glove"]), tuple(float(q) for q in raw["robot"]))


@dataclass(frozen=True)
class JoystickMap:
    joints: Mapping[str, JointMap]  # a subset of JOINTS; the rest follow the other method
    tuned: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.joints) - set(JOINTS)
        if unknown:
            raise JoystickError(f"joystick: unknown joints {sorted(unknown)}")
        for j, m in self.joints.items():
            m.check(j)
            if len(m.glove) != len(POINTS[j]):
                raise JoystickError(f"joystick {j}: {len(POINTS[j])} points expected, got {len(m.glove)}")

    def to_dict(self) -> dict:
        return {"tuned": self.tuned, "joints": {j: m.to_dict() for j, m in self.joints.items()}}

    @classmethod
    def from_dict(cls, raw: Mapping) -> JoystickMap:
        try:
            joints = {str(j): JointMap.from_dict(m) for j, m in (raw.get("joints") or {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise JoystickError(f"joystick: {exc!r}") from exc
        return cls(joints, str(raw.get("tuned", "")))


def default_joint(joint: str, medians: Mapping[str, Mapping[str, float]],
                  targets: Mapping[str, Mapping[str, float]], source: str | None = None) -> JointMap | None:
    """The joint's map from the calibration's example-pose medians (None: a pose or input is missing)."""
    source = source or SOURCE_CHOICES[joint][0]
    glove, robot = [], []
    for _, pose, _ in POINTS[joint]:
        x = raw_reading(source, medians.get(pose, {}))
        if x is None or pose not in targets or joint not in targets[pose]:
            return None
        glove.append(x)
        robot.append(float(targets[pose][joint]))
    return JointMap(source, tuple(glove), tuple(robot))


class JoystickRetargeter:
    """Glove signals -> the joystick joints (rad); the [켜기] reference take shifts the points."""

    def __init__(self, joystick: JoystickMap, home_rad: Mapping[str, float],
                 sweeps: FingerSweeps | None = None) -> None:
        self.joystick = joystick
        self.home_rad = dict(home_rad)
        self.sweeps = sweeps
        self.maps: dict[str, JointMap] = {}
        self.rest: dict[str, tuple[float, ...]] = {}
        self.reset()

    def reset(self) -> None:
        self.maps = dict(self.joystick.joints)
        self.rest = dict(self.sweeps.rest) if self.sweeps is not None else {}

    @property
    def joints(self) -> tuple[str, ...]:
        out = tuple(self.maps)
        return out + tuple(f for f, src in FOLLOWS.items() if src in self.maps)

    def corrected(self, joint: str) -> bool:
        """Whether the joint's reading has the sweeps' crosstalk taken off (its standard inputs)."""
        m = self.maps.get(joint)
        return (self.sweeps is not None and m is not None and joint in SWEEP_JOINT_INPUTS
                and source_inputs(m.source) == SWEEP_JOINT_INPUTS[joint])

    def reading(self, joint: str, signals: Mapping[str, float], source: str | None = None) -> float | None:
        src = source or self.maps[joint].source
        if self.sweeps is not None and joint in SWEEP_JOINT_INPUTS and source_inputs(src) == SWEEP_JOINT_INPUTS[joint]:
            return self.sweeps.corrected(joint, signals, self.rest)
        return raw_reading(src, signals)

    def ready(self, signals: Mapping[str, float]) -> bool:
        return all(self.reading(j, signals) is not None for j in self.maps)

    def solve(self, signals: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for j, m in self.maps.items():
            x = self.reading(j, signals)
            if x is None:
                raise KeyError(f"joystick: no {m.source} in the glove signals (check ready())")
            out[j] = m.angle(x)
        for f, src in FOLLOWS.items():
            if src in out:
                out[f] = out[src]
        return out

    def take_reference(self, samples: Sequence[Mapping[str, float]]) -> list[str]:
        """The operator holds the robot's home: shift each joint so that reading maps to home."""
        if self.sweeps is not None:
            for src in SOURCES:  # the crosstalk baseline, as flexion._take_rest
                values = [v for s in samples if (v := source_values(src, s)) is not None]
                if values:
                    now = tuple(statistics.median(v[i] for v in values) for i in range(len(values[0])))
                    if src == "thumb" or now[0] <= self.sweeps.rest[src][0] + 0.6:
                        self.rest[src] = now
        taken = []
        for j, m in list(self.maps.items()):
            values = [v for s in samples if (v := self.reading(j, s)) is not None]
            at_home = m.glove_at(self.home_rad[j])
            if not values or at_home is None:
                continue
            d = statistics.median(values) - at_home
            if abs(d) <= REFERENCE_TOL * abs(m.glove[-1] - m.glove[0]):
                self.maps[j] = m.shifted(d)
                taken.append(j)
        return taken

    def log_fields(self, signals: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for j, m in self.maps.items():
            x = self.reading(j, signals)
            if x is not None:
                out[f"joy_{j}"] = x
            out[f"joy_lo_{j}"], out[f"joy_hi_{j}"] = m.glove[0], m.glove[-1]
        return out
