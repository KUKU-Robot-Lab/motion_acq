"""Direct map (10.11 user: "switch to the other project", Nova2Dex): each robot joint from one SDK value.

Nova2Dex (github.com/Lvn-7/Nova2Dex-A-ROS-2-Framework-for-Dexterous-Hand-Teleoperation, MIT) drives an
Inspire hand from a Nova 2 with no per-operator calibration of its own: it reads the SenseGlove SDK
straight (here: motion_acq_nova2's nova2_sdk node, /motion_acq/nova2_sdk/<side>) and maps
* index / middle / ring / pinky: the SDK's normalized flexion (0 open .. 1 bent; the SDK learns its
  ranges while the operator moves the fingers, calibration state "Move Fingers" until then), read as a
  full bend at finger_full (Nova2Dex normalized_four_finger_flexion_max 0.928571), times the joint range;
* the thumb bend (RH56F1 thumb_2, Inspire thumb pitch): the thumb's normalized flexion, full at
  thumb_full (0.733333);
* the thumb rotation across the palm (RH56F1 thumb_1, Inspire thumb yaw): the SDK hand angle of the
  thumb's first joint about X, 0 rad (spread) .. -0.3 rad (across the palm) on the right glove
  (Nova2Dex / arm_hand_teleop); the left glove reads it mirrored (+).
The Nova 2 has one thumb bend sensor, so the glove's thumb mcp and pip move together (10.11 logs:
pip ~ f(mcp), R^2 0.91-0.98); one thumb value per robot thumb joint is all it can give.
The glove signals here are the SDK message's names with SDK_PREFIX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SDK_PREFIX = "sdk_"
FINGER_JOINTS = {"index_1": "index", "middle_1": "middle", "ring_1": "ring", "pinky_1": "pinky"}
THUMB_YAW = "thumb_0_x"  # SDK hand angle of the thumb's first joint, X axis (rad)
JOINTS = ("thumb_1", "thumb_2", *FINGER_JOINTS)


class DirectError(ValueError):
    pass


@dataclass(frozen=True)
class DirectConfig:
    finger_full: float = 0.928571  # normalized flexion read as a full finger bend
    thumb_full: float = 0.733333  # ... as a full thumb bend
    # thumb_0_x at the thumb spread away / across the palm, right glove (the left is mirrored)
    thumb_yaw_rad: tuple[float, float] = (0.0, -0.3)
    # robot thumb_1 at those two: spread (the open example, 0.8) .. across (thumb_opposed, 2.0)
    thumb_1_rad: tuple[float, float] = (0.8, 2.0)

    def __post_init__(self) -> None:
        if not (0.0 < self.finger_full <= 1.0 and 0.0 < self.thumb_full <= 1.0):
            raise DirectError("direct: finger_full and thumb_full must be in (0, 1]")
        if abs(self.thumb_yaw_rad[1] - self.thumb_yaw_rad[0]) < 1e-3:
            raise DirectError("direct: thumb_yaw_rad needs two different angles")


def direct_config(raw: Mapping | None) -> DirectConfig:
    raw = dict(raw or {})
    unknown = set(raw) - set(DirectConfig.__dataclass_fields__)
    if unknown:
        raise DirectError(f"direct: unknown keys {sorted(unknown)}")
    out = {}
    for k, v in raw.items():
        out[k] = (float(v[0]), float(v[1])) if k in ("thumb_yaw_rad", "thumb_1_rad") else float(v)
    return DirectConfig(**out)


def _unit(x: float) -> float:
    return min(max(x, 0.0), 1.0)


class DirectRetargeter:
    """SDK signals -> RH56F1 joint angles (rad), before the joint limits and amplitude."""

    def __init__(self, config: DirectConfig, limits_rad: Mapping[str, tuple[float, float]], side: str) -> None:
        missing = set(JOINTS) - set(limits_rad)
        if missing:
            raise DirectError(f"direct: the hand has no {sorted(missing)}")
        self.config = config
        self.limits_rad = dict(limits_rad)
        self.mirror = -1.0 if side == "left" else 1.0

    @staticmethod
    def inputs() -> list[str]:
        names = [f"nflex_{f}" for f in ("thumb", *FINGER_JOINTS.values())] + [THUMB_YAW]
        return [SDK_PREFIX + n for n in names]

    def ready(self, signals: Mapping[str, float]) -> bool:
        return all(n in signals for n in self.inputs())

    def units(self, signals: Mapping[str, float]) -> dict[str, float]:
        """0..1 per robot joint: 0 at the joint's open end (thumb_1: spread), 1 at its closed end."""
        c = self.config
        out = {j: _unit(float(signals[SDK_PREFIX + "nflex_" + f]) / c.finger_full) for j, f in FINGER_JOINTS.items()}
        out["thumb_2"] = _unit(float(signals[SDK_PREFIX + "nflex_thumb"]) / c.thumb_full)
        yaw = self.mirror * float(signals[SDK_PREFIX + THUMB_YAW])
        y0, y1 = c.thumb_yaw_rad
        out["thumb_1"] = _unit((yaw - y0) / (y1 - y0))
        return out

    def solve(self, signals: Mapping[str, float]) -> dict[str, float]:
        if not self.ready(signals):
            missing = [n for n in self.inputs() if n not in signals]
            raise DirectError(f"direct: no SDK values {missing} (is motion_acq_nova2 nova2_sdk running?)")
        u = self.units(signals)
        out = {}
        for j, x in u.items():
            if j == "thumb_1":
                a, b = self.config.thumb_1_rad
            else:
                a, b = self.limits_rad[j]
            out[j] = a + x * (b - a)
        return out

    def log_fields(self, signals: Mapping[str, float]) -> dict[str, float]:
        out = {n: float(signals[n]) for n in self.inputs() if n in signals}
        if self.ready(signals):
            out.update({f"unit_{j}": round(x, 4) for j, x in self.units(signals).items()})
        return out
