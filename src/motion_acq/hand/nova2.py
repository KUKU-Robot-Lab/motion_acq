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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PARTS = ("brake", "mcp", "pip", "dip")
SIDE_PREFIX = {"right": "r", "left": "l"}
SIDE_TAG = {"right": "rh", "left": "lh"}
GLOVES_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "hands" / "nova2_gloves.yaml"
_MAC = re.compile(r"[0-9A-F]{2}(:[0-9A-F]{2}){5}")
_SERIAL = re.compile(r"[0-9]+")


class GloveDataError(ValueError):
    pass


def check_serial(value: Any) -> str:
    """A Nova 2 serial is a digit string with its leading zeros ("00782").

    senseglove_ros finds the glove by this text, and the namespace carries it.
    An unquoted 00782 becomes 782 in YAML and 782.0 as a ROS -p parameter, so
    anything but a digit string is refused instead of silently renamed.
    """
    if not isinstance(value, str) or not _SERIAL.fullmatch(value):
        raise GloveDataError(
            f"glove serial must be a quoted digit string like \"00782\", not {value!r}"
            " (ROS: -p glove_serial:=\"'00782'\" or pass glove_topic)"
        )
    return value


def glove_topic(serial: str, side: str) -> str:
    """senseglove_ros state topic: hardware.launch.py namespace
    /senseglove/glove<serial>/<rh|lh> + senseglove_state_broadcaster topic_name."""
    return f"/senseglove/glove{check_serial(serial)}/{SIDE_TAG[side]}/senseglove_states"


@dataclass(frozen=True)
class Nova2Glove:
    side: str
    serial: str
    mac: str  # BLE address (firmware v2): BlueZ trusts it, no pairing or bonding
    name: str  # advertised name, "Nova 2-<serial>-<R|L>"

    @property
    def topic(self) -> str:
        return glove_topic(self.serial, self.side)


def load_gloves(path: Path = GLOVES_CONFIG) -> dict[str, Nova2Glove]:
    """The lab's Nova 2 gloves by side (configs/hands/nova2_gloves.yaml)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    gloves = {}
    for side, entry in (raw.get("gloves") or {}).items():
        if side not in SIDE_TAG:
            raise GloveDataError(f"{path}: glove side must be right or left, not {side!r}")
        glove = Nova2Glove(side=side, serial=check_serial(entry.get("serial")),
                           mac=str(entry.get("mac", "")).upper(), name=str(entry.get("name", "")))
        if not _MAC.fullmatch(glove.mac):
            raise GloveDataError(f"{path}: {side} glove mac {glove.mac!r} is not AA:BB:CC:DD:EE:FF")
        expected = f"Nova 2-{glove.serial}-{side[0].upper()}"
        if glove.name != expected:
            raise GloveDataError(f"{path}: {side} glove name {glove.name!r}, expected {expected!r}")
        gloves[side] = glove
    return gloves


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


# Thumb-to-finger tip distances from SenseGloveState.finger_tip_position (thumb first;
# senseglove_ros fills it with the SGCore hand model's distal joint positions, mm). They
# join the angle dict under these keys so a feature can use them (pinch_* features).
TIP_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_DIST_KEYS = tuple(f"tipdist_{f}" for f in TIP_FINGERS[1:])


def tip_distances(tips: Sequence[Sequence[float]]) -> dict[str, float]:
    """{tipdist_<finger>: |thumb tip - finger tip|}; {} when the glove sends no usable tips."""
    if len(tips) != len(TIP_FINGERS):
        return {}
    points = [tuple(float(v) for v in tip) for tip in tips]
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points):
        return {}
    if all(v == 0.0 for p in points for v in p):  # not filled (fake glove, old driver)
        return {}
    thumb = points[0]
    return {key: math.dist(thumb, p) for key, p in zip(TIP_DIST_KEYS, points[1:])}


def features(angles: Mapping[str, float], specs: Sequence[FeatureSpec]) -> dict[str, float]:
    """Feature values; a feature whose inputs are absent (tip distances without tip data) is left out."""
    return {spec.name: spec.value(angles) for spec in specs if all(j in angles for j in spec.weights)}
