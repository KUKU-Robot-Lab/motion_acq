"""Direct map (10.11 user: "switch to the other project", Nova2Dex): each robot joint from one SDK value.

Nova2Dex (github.com/Lvn-7/Nova2Dex-A-ROS-2-Framework-for-Dexterous-Hand-Teleoperation, MIT) drives an
Inspire hand from a Nova 2 with no per-operator calibration of its own: it reads the SenseGlove SDK
straight (here: motion_acq_nova2's nova2_sdk node, /motion_acq/nova2_sdk/<side>) and maps
* index / middle / ring / pinky: the SDK's normalized flexion (0..1; the SDK learns its ranges while
  the operator moves the fingers, calibration state "Move Fingers" until then), the joint's range from
  finger_open to finger_full (Nova2Dex: 0 .. 0.928571);
* the thumb bend (RH56F1 thumb_2, Inspire thumb pitch): the thumb's normalized flexion, thumb_open ..
  thumb_full (Nova2Dex: 0 .. 0.733333);
* the thumb rotation across the palm (RH56F1 thumb_1, Inspire thumb yaw): the SDK hand angle of the
  thumb's first joint about X, from thumb_yaw_rad[side][0] to [1] (Nova2Dex right glove: 0 .. -0.3 rad)
  onto thumb_1_rad.
The Nova 2 has one thumb bend sensor, so the glove's thumb mcp and pip move together (10.11 logs:
pip ~ f(mcp), R^2 0.91-0.98); one thumb value per robot thumb joint is all it can give.

10.11 recording (logs/nova2_sdk/sdk_20261011_204233 on arm4090, both hands, "All Sensors Moved"): the
open hand reads 0.14-0.33 normalized flexion, not 0, so the configured open/full ends are those medians;
thumb_0_x is beside the index 0.12 / -0.07 and across the palm -0.08 / 0.19 (right / left), the spread
thumb only 0.04 / 0.09 past beside, so the robot thumb turns from home (beside) to across, not out.
The glove signals here are the SDK message's names with SDK_PREFIX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

SDK_PREFIX = "sdk_"
FINGER_JOINTS = {"index_1": "index", "middle_1": "middle", "ring_1": "ring", "pinky_1": "pinky"}
FINGERS = tuple(FINGER_JOINTS.values())
THUMB_YAW = "thumb_0_x"  # SDK hand angle of the thumb's first joint, X axis (rad)
JOINTS = ("thumb_1", "thumb_2", *FINGER_JOINTS)
SIDES = ("right", "left")


class DirectError(ValueError):
    pass


def _per_finger(value: float) -> dict[str, float]:
    return {f: float(value) for f in FINGERS}


@dataclass(frozen=True)
class DirectConfig:
    """Defaults are Nova2Dex's numbers; configs/hands/nova2_to_rh56f1.yaml has the measured ones."""

    finger_open: Mapping[str, float] = field(default_factory=lambda: _per_finger(0.0))
    finger_full: Mapping[str, float] = field(default_factory=lambda: _per_finger(0.928571))
    thumb_open: float = 0.0
    thumb_full: float = 0.733333
    # side -> thumb_0_x at the thumb_1_rad ends
    thumb_yaw_rad: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {"right": (0.0, -0.3), "left": (0.0, 0.3)})
    thumb_1_rad: tuple[float, float] = (0.8, 2.0)

    def __post_init__(self) -> None:
        if set(self.finger_open) != set(FINGERS) or set(self.finger_full) != set(FINGERS):
            raise DirectError(f"direct: finger_open and finger_full need {list(FINGERS)}")
        for f in FINGERS:
            if not self.finger_full[f] - self.finger_open[f] >= 0.05:
                raise DirectError(f"direct: {f} finger_full must be at least 0.05 above finger_open")
        if not self.thumb_full - self.thumb_open >= 0.05:
            raise DirectError("direct: thumb_full must be at least 0.05 above thumb_open")
        if set(self.thumb_yaw_rad) != set(SIDES):
            raise DirectError("direct: thumb_yaw_rad needs right and left")
        for side, (a, b) in self.thumb_yaw_rad.items():
            if abs(b - a) < 0.02:
                raise DirectError(f"direct: thumb_yaw_rad {side} needs two angles at least 0.02 rad apart")


def direct_config(raw: Mapping | None) -> DirectConfig:
    raw = dict(raw or {})
    unknown = set(raw) - set(DirectConfig.__dataclass_fields__)
    if unknown:
        raise DirectError(f"direct: unknown keys {sorted(unknown)}")
    out: dict = {}
    for k in ("finger_open", "finger_full"):
        if k in raw:
            v = raw[k]
            out[k] = {str(f): float(x) for f, x in v.items()} if isinstance(v, Mapping) else _per_finger(v)
    for k in ("thumb_open", "thumb_full"):
        if k in raw:
            out[k] = float(raw[k])
    if "thumb_yaw_rad" in raw:
        out["thumb_yaw_rad"] = {str(s): (float(v[0]), float(v[1])) for s, v in raw["thumb_yaw_rad"].items()}
    if "thumb_1_rad" in raw:
        out["thumb_1_rad"] = (float(raw["thumb_1_rad"][0]), float(raw["thumb_1_rad"][1]))
    return DirectConfig(**out)


def _unit(x: float, a: float, b: float) -> float:
    """0 at a, 1 at b, clipped."""
    return min(max((x - a) / (b - a), 0.0), 1.0)


class DirectRetargeter:
    """SDK signals -> RH56F1 joint angles (rad), before the joint limits and amplitude."""

    def __init__(self, config: DirectConfig, limits_rad: Mapping[str, tuple[float, float]], side: str) -> None:
        missing = set(JOINTS) - set(limits_rad)
        if missing:
            raise DirectError(f"direct: the hand has no {sorted(missing)}")
        if side not in SIDES:
            raise DirectError(f"direct: side must be right or left, not {side!r}")
        self.config = config
        self.limits_rad = dict(limits_rad)
        self.yaw = config.thumb_yaw_rad[side]

    @staticmethod
    def inputs() -> list[str]:
        names = [f"nflex_{f}" for f in ("thumb", *FINGERS)] + [THUMB_YAW]
        return [SDK_PREFIX + n for n in names]

    def ready(self, signals: Mapping[str, float]) -> bool:
        return all(n in signals for n in self.inputs())

    def units(self, signals: Mapping[str, float]) -> dict[str, float]:
        """0..1 per robot joint: 0 at the joint's open end (thumb_1: thumb_1_rad[0]), 1 at its closed end."""
        c = self.config
        out = {j: _unit(float(signals[SDK_PREFIX + "nflex_" + f]), c.finger_open[f], c.finger_full[f])
               for j, f in FINGER_JOINTS.items()}
        out["thumb_2"] = _unit(float(signals[SDK_PREFIX + "nflex_thumb"]), c.thumb_open, c.thumb_full)
        out["thumb_1"] = _unit(float(signals[SDK_PREFIX + THUMB_YAW]), *self.yaw)
        return out

    def solve(self, signals: Mapping[str, float]) -> dict[str, float]:
        if not self.ready(signals):
            missing = [n for n in self.inputs() if n not in signals]
            raise DirectError(f"direct: no SDK values {missing} (is motion_acq_nova2 nova2_sdk running?)")
        out = {}
        for j, x in self.units(signals).items():
            a, b = self.config.thumb_1_rad if j == "thumb_1" else self.limits_rad[j]
            out[j] = a + x * (b - a)
        return out

    def log_fields(self, signals: Mapping[str, float]) -> dict[str, float]:
        out = {n: float(signals[n]) for n in self.inputs() if n in signals}
        if self.ready(signals):
            out.update({f"unit_{j}": round(x, 4) for j, x in self.units(signals).items()})
        return out
