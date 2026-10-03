"""Synthetic Nova 2 angles for fake gloves and tests (no glove needed).

A virtual operator whose glove reads OPEN at curl = 0 and FIST at curl = 1;
thumb_opposition moves the thumb CMC flexion from its open to its opposed
value. Values are plausible radians, not measurements.
"""

from __future__ import annotations

from motion_acq.hand.nova2 import FINGERS, PARTS, SIDE_PREFIX

OPEN = {"brake": 0.05, "mcp": 0.05, "pip": 0.05, "dip": 0.03}
FIST = {"brake": 0.02, "mcp": 1.40, "pip": 1.50, "dip": 0.90}
THUMB_OPEN = {"brake": 0.30, "mcp": 0.00, "pip": 0.05, "dip": 0.05}
THUMB_BENT = {"pip": 0.60, "dip": 0.50}
THUMB_OPPOSED_MCP = 0.90


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def synthetic_angles(curl: float, thumb_bend: float, thumb_opposition: float) -> dict[str, float]:
    """Unprefixed glove angles ("index_mcp" -> rad)."""
    angles = {}
    for finger in FINGERS[1:]:
        for part in PARTS:
            angles[f"{finger}_{part}"] = _lerp(OPEN[part], FIST[part], curl)
    angles["thumb_brake"] = THUMB_OPEN["brake"]
    angles["thumb_mcp"] = _lerp(THUMB_OPEN["mcp"], THUMB_OPPOSED_MCP, thumb_opposition)
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
