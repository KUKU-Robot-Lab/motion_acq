"""Synthetic Nova 2 signals for fake gloves and tests (no glove needed).

A virtual operator: each finger reads OPEN at curl 0 and FIST at curl 1; the thumb has
a bend, an opposition (raises thumb_brake) and a "spread" (0 = beside the index, 1 = far
out); tip distances shrink as a finger curls and as the thumb opposes toward it, and go
to TIP_TOUCH for a pinch. Like the real glove, some signals are coupled: curling the
ring finger raises thumb_brake a little (10.05 left log, corr 0.41). Ends of the joints
are near bumsu's Nova 2 calibration of 2026-10-02 (pip 0.07 -> 1.74, thumb_pip 0.01 ->
0.72, thumb_brake 0.00 -> 1.05 rad); the rest are plausible values, not measurements.
"""

from __future__ import annotations

from typing import Mapping

from motion_acq.hand.nova2 import FINGERS, PARTS, SIDE_PREFIX, THUMB_TIP_KEYS, TIP_DIST_KEYS

OPEN = {"brake": 0.0, "mcp": 0.05, "pip": 0.07, "dip": 0.07}
FIST = {"brake": 0.0, "mcp": 1.40, "pip": 1.74, "dip": 1.74}
THUMB_OPEN = {"brake": 0.00, "mcp": 0.00, "pip": 0.01, "dip": 0.01}
THUMB_BENT = {"pip": 0.72, "dip": 0.72}
THUMB_OPPOSED = {"brake": 1.05, "mcp": 0.40}
SPREAD_BRAKE = -0.35  # thumb spread far out lowers thumb_brake
RING_TO_THUMB = 0.15  # coupling: ring curl leaks into thumb_brake
TIP_OPEN = {"index": 90.0, "middle": 100.0, "ring": 105.0, "pinky": 110.0}  # mm
TIP_CURL = {"index": 40.0, "middle": 45.0, "ring": 50.0, "pinky": 55.0}  # shrink by curl 1
TIP_OPPOSE = {"index": 15.0, "middle": 30.0, "ring": 40.0, "pinky": 40.0}  # shrink by opposition 1
TIP_TOUCH = 15.0
CURLED = ("index", "middle", "ring", "pinky")


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def synthetic_angles(curl: float | Mapping[str, float], thumb_bend: float, thumb_opposition: float,
                     pinch: tuple[str, float] | None = None, spread: float = 0.0) -> dict[str, float]:
    """Unprefixed glove signals ("index_mcp" -> rad, "tipdist_index" -> mm).

    curl: one value for all four fingers or {finger: curl} (missing = 0; pinky follows ring
    unless given). pinch = (finger, amount): that finger's tip distance -> TIP_TOUCH."""
    if isinstance(curl, Mapping):
        curls = {f: float(curl.get(f, curl.get("ring", 0.0) if f == "pinky" else 0.0)) for f in CURLED}
    else:
        curls = {f: float(curl) for f in CURLED}
    angles = {}
    for finger in CURLED:
        for part in PARTS:
            angles[f"{finger}_{part}"] = _lerp(OPEN[part], FIST[part], curls[finger])
    angles["thumb_brake"] = (_lerp(THUMB_OPEN["brake"], THUMB_OPPOSED["brake"], thumb_opposition)
                             + SPREAD_BRAKE * spread + RING_TO_THUMB * curls["ring"])
    angles["thumb_mcp"] = _lerp(THUMB_OPEN["mcp"], THUMB_OPPOSED["mcp"], thumb_opposition)
    angles["thumb_pip"] = _lerp(THUMB_OPEN["pip"], THUMB_BENT["pip"], thumb_bend)
    angles["thumb_dip"] = angles["thumb_pip"]
    for key, finger in zip(TIP_DIST_KEYS, FINGERS[1:]):
        d = TIP_OPEN[finger] - TIP_CURL[finger] * curls[finger] - TIP_OPPOSE[finger] * thumb_opposition
        angles[key] = max(d + 10.0 * spread, TIP_TOUCH)
    if pinch is not None:
        finger, amount = pinch
        key = f"tipdist_{finger}"
        angles[key] = _lerp(angles[key], TIP_TOUCH, amount)
    # thumb tip (mm, hand frame): swings across the palm with opposition, out with spread
    tip = (20.0 + 40.0 * thumb_opposition - 30.0 * spread, 60.0 - 50.0 * thumb_opposition + 20.0 * spread,
           30.0 * thumb_bend)
    angles.update(zip(THUMB_TIP_KEYS, tip))
    return angles


def synthetic_tips(angles: Mapping[str, float]) -> list[tuple[float, float, float]]:
    """finger_tip_position (thumb first) whose thumb-to-finger distances are the tipdist_* values."""
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
    thumb = tuple(angles.get(k, 0.0) for k in THUMB_TIP_KEYS)
    tips = [thumb]
    for key, axis in zip(TIP_DIST_KEYS, axes):
        d = angles.get(key, 0.0)
        tips.append((thumb[0] + axis[0] * d, thumb[1] + axis[1] * d, thumb[2] + axis[2] * d))
    return tips


def synthetic_state(side: str, angles: Mapping[str, float]) -> tuple[list[str], list[float]]:
    """(joint_names, position) as senseglove_ros publishes them for one glove."""
    prefix = SIDE_PREFIX[side]
    names = [f"{prefix}_{finger}_{part}" for finger in FINGERS for part in PARTS]
    return names, [angles[n[len(prefix) + 1:]] for n in names]


# The example poses of configs/hands/nova2_to_rh56f1.yaml, as the virtual operator does them.
POSE_ANGLES = {
    "open": synthetic_angles(0.0, 0.0, 0.0, spread=1.0),
    "flat": synthetic_angles(0.0, 0.0, 0.0),
    "fist": synthetic_angles(1.0, 1.0, 0.0),
    "index": synthetic_angles({"index": 1.0}, 0.0, 0.0),
    "middle": synthetic_angles({"middle": 1.0}, 0.0, 0.0),
    "ring": synthetic_angles({"ring": 1.0}, 0.0, 0.0),
    "thumb_bend": synthetic_angles(0.0, 1.0, 0.0),
    "thumb_opposed": synthetic_angles(0.0, 0.3, 1.0),
    "pinch_index": synthetic_angles({"index": 0.45}, 0.5, 0.35, ("index", 1.0)),
    "pinch_middle": synthetic_angles({"middle": 0.45}, 0.4, 0.65, ("middle", 1.0)),
    "pinch_ring": synthetic_angles({"ring": 0.5}, 0.35, 0.9, ("ring", 1.0)),
}
