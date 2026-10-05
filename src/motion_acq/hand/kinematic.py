"""Kinematic retargeting: the glove's hand model -> RH56F1 joints by matching fingertips.

10.06 user: retarget the Nova 2 hand onto the RH56F1 instead of calibrating signal by
signal. Nova 2 (SGCore) streams a hand model: 4 positions per finger (hand_position,
finger * 4 + joint, joint 0 = knuckle) and the five fingertips (finger_tip_position),
mm in the glove's hand frame. The RH56F1 tip positions come from its vendor URDF
(motion_acq.hand.fk), tabulated per finger (one joint each) and for the thumb (two
joints) in configs/hands/rh56f1_tips_<side>.npz (scripts/make_rh56f1_tip_tables.py).

Alignment (per user, from the open hand, 2 s): a similarity transform (rotation or
reflection, scale, offset) that puts the operator's four knuckles and four straight
fingertips onto the robot's (Umeyama). It absorbs the glove frame convention and the
hand size.

Each tick (DexPilot-style, closed form over the tables):
* each finger joint: the robot tip closest to the operator's aligned tip;
* the thumb (rotation, bend): the table point closest to the operator's aligned thumb tip,
  plus, for every finger the thumb nears, the thumb-to-finger vector; as the operator's
  tips touch (aligned distance below touch_m) that vector's target shrinks to zero, so a
  pinch puts the robot tips together. The pinched finger is then re-solved with the same
  term. A small term toward the previous solution keeps the grid from flickering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FINGERS = ("index", "middle", "ring", "pinky")
FINGER_JOINT = {"index": "index_1", "middle": "middle_1", "ring": "ring_1", "pinky": "pinky_1"}
GLOVE_FINGER = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
TABLES = Path(__file__).resolve().parents[3] / "configs" / "hands"


def table_path(side: str) -> Path:
    return TABLES / f"rh56f1_tips_{side}.npz"


def point_keys(prefix: str) -> tuple[str, str, str]:
    return (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")


def knuckle_prefix(finger: str) -> str:
    return f"knuckle_{finger}"


def tip_prefix(finger: str) -> str:
    return f"tip_{finger}"


def glove_points(hand_position: Sequence[Sequence[float]], tips: Sequence[Sequence[float]]) -> dict[str, float]:
    """SenseGloveState hand_position (20) and finger_tip_position (5) -> flat signals
    (knuckle_<finger>_x.., tip_<finger>_x..); {} when the glove sends no hand model."""
    if len(hand_position) < 20 or len(tips) != 5:
        return {}
    pts = np.asarray([tuple(p) for p in tips], float)
    knuckles = np.asarray([tuple(hand_position[GLOVE_FINGER[f] * 4]) for f in FINGERS], float)
    if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(knuckles))) or not np.any(knuckles) or not np.any(pts):
        return {}
    out = {}
    for f, k in zip(FINGERS, knuckles):
        out.update(zip(point_keys(knuckle_prefix(f)), map(float, k)))
    for f, p in zip(("thumb",) + FINGERS, pts):
        out.update(zip(point_keys(tip_prefix(f)), map(float, p)))
    return out


def points_from_signals(signals: Mapping[str, float]) -> dict[str, np.ndarray] | None:
    names = [knuckle_prefix(f) for f in FINGERS] + [tip_prefix(f) for f in ("thumb",) + FINGERS]
    try:
        return {n: np.array([signals[k] for k in point_keys(n)], float) for n in names}
    except KeyError:
        return None


@dataclass(frozen=True)
class Alignment:
    """Glove hand frame (mm) -> robot hand base frame (m): x_r = scale * R @ x_g + offset."""
    rotation: np.ndarray  # 3x3, orthogonal (det +-1: the glove frame may be left-handed)
    scale: float
    offset: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.scale * (self.rotation @ x) + self.offset

    def to_dict(self) -> dict:
        return {"rotation": self.rotation.tolist(), "scale": self.scale, "offset": self.offset.tolist()}

    @classmethod
    def from_dict(cls, raw: Mapping) -> Alignment:
        r = np.asarray(raw["rotation"], float)
        o = np.asarray(raw["offset"], float)
        s = float(raw["scale"])
        if r.shape != (3, 3) or o.shape != (3,) or not np.isfinite(s) or s <= 0:
            raise ValueError("bad kinematic alignment")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6):
            raise ValueError("kinematic alignment rotation is not orthogonal")
        return cls(r, s, o)


def umeyama(src: np.ndarray, dst: np.ndarray) -> Alignment:
    """Similarity transform (orthogonal incl. reflection, scale, offset) taking src onto dst."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a)
    rot = u @ vt  # best orthogonal map; may be a reflection (left-handed glove frame)
    scale = float(sig.sum() / np.sum(a ** 2))
    return Alignment(rot, scale, mu_d - scale * rot @ mu_s)


@dataclass(frozen=True)
class TipTables:
    finger_q: np.ndarray  # (n,)
    finger_tips: Mapping[str, np.ndarray]  # finger -> (n, 3) m
    thumb_q: np.ndarray  # (m, 2): thumb_1, thumb_2
    thumb_tips: np.ndarray  # (m, 3)
    knuckles: Mapping[str, np.ndarray]  # finger -> (3,)

    @classmethod
    def load(cls, path: Path) -> TipTables:
        d = np.load(path)
        return cls(d["finger_q"], {f: d[f"tip_{f}"] for f in FINGERS}, d["thumb_q"], d["thumb_tips"],
                   {f: d[f"knuckle_{f}"] for f in FINGERS})

    def open_points(self) -> np.ndarray:
        """Robot knuckles and straight fingertips (the alignment's targets)."""
        return np.array([self.knuckles[f] for f in FINGERS] + [self.finger_tips[f][0] for f in FINGERS])


def fit_alignment(tables: TipTables, open_points: Mapping[str, np.ndarray]) -> Alignment:
    src = np.array([open_points[knuckle_prefix(f)] for f in FINGERS] + [open_points[tip_prefix(f)] for f in FINGERS])
    return umeyama(src, tables.open_points())


@dataclass(frozen=True)
class KinematicConfig:
    near_m: float = 0.045  # operator thumb-to-finger distance (robot scale) where the pinch term starts
    touch_m: float = 0.015  # ... and where the target becomes "tips together"
    pinch_weight: float = 8.0
    smooth_weight: float = 2e-4  # per rad^2, toward the previous solution


class KinematicRetargeter:
    def __init__(self, tables: TipTables, alignment: Alignment, config: KinematicConfig = KinematicConfig()) -> None:
        self.t, self.a, self.c = tables, alignment, config
        self.prev: dict[str, float] | None = None

    def _finger(self, f: str, target: np.ndarray, extra: tuple[np.ndarray, float] | None) -> int:
        tips = self.t.finger_tips[f]
        cost = np.sum((tips - target) ** 2, axis=1)
        if extra is not None:
            want, w = extra
            cost = cost + w * np.sum((tips - want) ** 2, axis=1)
        if self.prev is not None:
            cost = cost + self.c.smooth_weight * (self.t.finger_q - self.prev[FINGER_JOINT[f]]) ** 2
        return int(np.argmin(cost))

    def solve(self, points: Mapping[str, np.ndarray]) -> dict[str, float]:
        c = self.c
        tgt = {f: self.a.apply(points[tip_prefix(f)]) for f in ("thumb",) + FINGERS}
        idx = {f: self._finger(f, tgt[f], None) for f in FINGERS}
        tips = {f: self.t.finger_tips[f][idx[f]] for f in FINGERS}
        # thumb: its own tip plus the pinch vectors of the fingers it nears
        cost = np.sum((self.t.thumb_tips - tgt["thumb"]) ** 2, axis=1)
        pinch = {}
        for f in FINGERS:
            d = float(np.linalg.norm(tgt[f] - tgt["thumb"]))
            w = c.pinch_weight * min(max((c.near_m - d) / (c.near_m - c.touch_m), 0.0), 1.0)
            if w <= 0.0:
                continue
            shrink = min(max((d - c.touch_m) / (c.near_m - c.touch_m), 0.0), 1.0)
            vec = shrink * (tgt[f] - tgt["thumb"])  # thumb -> finger; zero once the operator's tips touch
            pinch[f] = (vec, w)
            cost = cost + w * np.sum((tips[f] - self.t.thumb_tips - vec) ** 2, axis=1)
        if self.prev is not None:
            cost = cost + c.smooth_weight * np.sum(
                (self.t.thumb_q - np.array([self.prev["thumb_1"], self.prev["thumb_2"]])) ** 2, axis=1)
        k = int(np.argmin(cost))
        thumb_tip = self.t.thumb_tips[k]
        for f, (vec, w) in pinch.items():  # the pinched finger meets the thumb too
            idx[f] = self._finger(f, tgt[f], (thumb_tip + vec, w))
        q = {"thumb_1": float(self.t.thumb_q[k, 0]), "thumb_2": float(self.t.thumb_q[k, 1])}
        q.update({FINGER_JOINT[f]: float(self.t.finger_q[idx[f]]) for f in FINGERS})
        self.prev = q
        return q
