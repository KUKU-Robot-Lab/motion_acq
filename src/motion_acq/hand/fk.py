"""RH56F1 fingertip forward kinematics from the vendor URDF (numpy, no ROS).

Used to find and check the pinch poses (configs/hands/nova2_to_rh56f1.yaml pinch.targets):
the thumb tip meets a finger tip only for one combination of thumb rotation, thumb bend and
finger curl. Joints use the hand map names (thumb_1, thumb_2, index_1, ..., pinky_1); the
distal joints follow through the URDF mimic tags.

    python -m motion_acq.hand.fk ~/rl_ws/urdf/vendor/RH56F1/RH56F1_R/urdf/RH56F1_R.urdf
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

URDF_FINGER = {"index": "index", "middle": "middle", "ring": "ring", "pinky": "little", "thumb": "thumb"}


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, t: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * k + (1.0 - np.cos(t)) * k @ k


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    revolute: bool
    origin: np.ndarray  # 4x4
    axis: np.ndarray | None
    mimic: tuple[str, float, float] | None


class TipFk:
    def __init__(self, urdf: Path) -> None:
        root = ET.parse(str(urdf)).getroot()
        self.joints: dict[str, _Joint] = {}  # by child link
        for j in root.iter("joint"):  # type: ignore[assignment]
            o = j.find("origin")
            xyz = [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
            rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            origin = np.eye(4)
            origin[:3, :3] = _rpy(*rpy)
            origin[:3, 3] = xyz
            a, m = j.find("axis"), j.find("mimic")
            self.joints[j.find("child").get("link")] = _Joint(
                j.get("name"), j.find("parent").get("link"), j.get("type") == "revolute", origin,
                np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None,
                (m.get("joint"), float(m.get("multiplier")), float(m.get("offset") or 0.0)) if m is not None else None)
        self.by_name = {j.name: j for j in self.joints.values()}
        names = set(self.by_name)
        self.prefix = next(p for p in ("right_", "left_") if f"{p}thumb_1_joint" in names)

    def _urdf_q(self, q: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for joint, value in q.items():
            finger, _, n = joint.rpartition("_")
            out[f"{self.prefix}{URDF_FINGER[finger]}_{n}_joint"] = float(value)
        return out

    def _angle(self, name: str, uq: Mapping[str, float]) -> float:
        """A joint's angle; a mimic joint follows its source (chains: thumb_4 <- thumb_3 <- thumb_2)."""
        j = self.by_name[name]
        if j.mimic is None:
            return uq.get(name, 0.0)
        source, mul, off = j.mimic
        return self._angle(source, uq) * mul + off

    def link_position(self, q: Mapping[str, float], link: str) -> np.ndarray:
        uq = self._urdf_q(q)
        chain = []
        while link in self.joints:
            chain.append(self.joints[link])
            link = self.joints[link].parent
        t = np.eye(4)
        for j in reversed(chain):
            t = t @ j.origin
            if j.revolute:
                angle = self._angle(j.name, uq)
                r = np.eye(4)
                r[:3, :3] = _axis_angle(j.axis, angle)
                t = t @ r
        return t[:3, 3]

    def tip_distance(self, q: Mapping[str, float], finger: str) -> float:
        """Thumb tip to finger tip (m) at joint values q (missing joints = 0)."""
        return float(np.linalg.norm(self.link_position(q, "thumb_tip")
                                    - self.link_position(q, f"{URDF_FINGER[finger]}_tip")))


def best_pinch(fk: TipFk, finger: str, steps: int = 40) -> tuple[float, dict[str, float]]:
    """Coarse grid search for the closest thumb-tip / finger-tip pose (distance m, joints)."""
    best = (np.inf, {})
    for t1 in np.linspace(1.2, 2.0944, steps):
        for t2 in np.linspace(0.0, 0.474555, steps // 2):
            for f in np.linspace(0.3, 1.2, steps // 2):
                q = {"thumb_1": t1, "thumb_2": t2, f"{finger}_1": f}
                d = fk.tip_distance(q, finger)
                if d < best[0]:
                    best = (d, q)
    return best


def main(argv: list[str]) -> int:
    fk = TipFk(Path(argv[1]))
    for finger in ("index", "middle", "ring", "pinky"):
        d, q = best_pinch(fk, finger)
        print(f"{finger:7s} {d * 1000:5.1f} mm  " + "  ".join(f"{k} {v:.3f}" for k, v in q.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
