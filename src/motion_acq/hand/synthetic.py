"""Synthetic Nova 2 angles for fake gloves and tests (no glove needed).

A virtual operator whose glove reads OPEN at curl = 0 and FIST at curl = 1;
thumb_opposition raises thumb_brake from its open to its opposed value. Ends
of the joints the retargeting reads (finger pip, thumb_pip, thumb_brake) are
near bumsu's Nova 2 calibration of 2026-10-02 (pip 0.07 -> 1.74, thumb_pip
0.01 -> 0.72, thumb_brake 0.00 -> 1.05 rad); the other joints are plausible
radians, not measurements.
"""

from __future__ import annotations

from motion_acq.hand.nova2 import FINGERS, PARTS, SIDE_PREFIX

OPEN = {"brake": 0.05, "mcp": 0.05, "pip": 0.07, "dip": 0.03}
FIST = {"brake": 0.02, "mcp": 1.40, "pip": 1.74, "dip": 0.90}
THUMB_OPEN = {"brake": 0.00, "mcp": 0.00, "pip": 0.01, "dip": 0.05}
THUMB_BENT = {"pip": 0.72, "dip": 0.50}
THUMB_OPPOSED = {"brake": 1.05, "mcp": 0.40}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def synthetic_angles(curl: float, thumb_bend: float, thumb_opposition: float) -> dict[str, float]:
    """Unprefixed glove angles ("index_mcp" -> rad)."""
    angles = {}
    for finger in FINGERS[1:]:
        for part in PARTS:
            angles[f"{finger}_{part}"] = _lerp(OPEN[part], FIST[part], curl)
    angles["thumb_brake"] = _lerp(THUMB_OPEN["brake"], THUMB_OPPOSED["brake"], thumb_opposition)
    angles["thumb_mcp"] = _lerp(THUMB_OPEN["mcp"], THUMB_OPPOSED["mcp"], thumb_opposition)
    angles["thumb_pip"] = _lerp(THUMB_OPEN["pip"], THUMB_BENT["pip"], thumb_bend)
    angles["thumb_dip"] = _lerp(THUMB_OPEN["dip"], THUMB_BENT["dip"], thumb_bend)
    return angles


def synthetic_state(side: str, angles: dict[str, float]) -> tuple[list[str], list[float]]:
    """(joint_names, position) as senseglove_ros publishes them for one glove."""
    prefix = SIDE_PREFIX[side]
    names = [f"{prefix}_{finger}_{part}" for finger in FINGERS for part in PARTS]
    return names, [angles[n[len(prefix) + 1:]] for n in names]


POSE_ANGLES = {
    "open": synthetic_angles(0.0, 0.0, 0.0),
    "fist": synthetic_angles(1.0, 1.0, 0.0),
    "thumb_opposed": synthetic_angles(0.0, 0.0, 1.0),
}
