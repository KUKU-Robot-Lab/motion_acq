"""SenseGlove Nova 2 joint angles (senseglove_ros SenseGloveState) -> features.

senseglove_ros (humble/jazzy) publishes, per glove, 20 joints
``<p>_<finger>_{brake,mcp,pip,dip}`` with p = r|l and finger in
thumb, index, middle, ring, pinky (radians). In senseglove_robot.cpp
updateJointPositions, sub-index 0 (named ``*_brake``) carries the negated Z
(abduction) of the first joint and mcp/pip/dip carry the Y (flexion) of the
three joints. That reading of the field names is from the driver source and
must be confirmed on the real glove.

A *feature* is a weighted sum of these angles (configs/hands/nova2_to_rh56f1.yaml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PARTS = ("brake", "mcp", "pip", "dip")
SIDE_PREFIX = {"right": "r", "left": "l"}


class GloveDataError(ValueError):
    pass


def glove_joint_names(side: str) -> tuple[str, ...]:
    prefix = SIDE_PREFIX[side]
    return tuple(f"{prefix}_{finger}_{part}" for finger in FINGERS for part in PARTS)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    weights: Mapping[str, float]  # unprefixed joint ("index_mcp") -> weight

    def value(self, angles: Mapping[str, float]) -> float:
        return float(sum(w * angles[j] for j, w in self.weights.items()))


def angles_from_state(
    joint_names: Sequence[str], positions: Sequence[float], side: str
) -> dict[str, float]:
    """Unprefixed glove angles ("index_mcp" -> rad) for one glove side."""
    if len(joint_names) != len(positions):
        raise GloveDataError("joint_names and position differ in length")
    prefix = SIDE_PREFIX[side] + "_"
    angles = {}
    for name, value in zip(joint_names, positions, strict=True):
        if not name.startswith(prefix):
            raise GloveDataError(f"{name!r} is not a {side} glove joint")
        value = float(value)
        if not math.isfinite(value):
            raise GloveDataError(f"non-finite {name}")
        angles[name[len(prefix):]] = value
    missing = [j for j in (n[len(prefix):] for n in glove_joint_names(side)) if j not in angles]
    if missing:
        raise GloveDataError(f"missing glove joints: {missing}")
    return angles


def features(angles: Mapping[str, float], specs: Sequence[FeatureSpec]) -> dict[str, float]:
    return {spec.name: spec.value(angles) for spec in specs}
