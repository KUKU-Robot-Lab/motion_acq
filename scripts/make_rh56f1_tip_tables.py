#!/usr/bin/env python3
"""RH56F1 fingertip tables for the kinematic retargeting (motion_acq.hand.kinematic).

    .venv/bin/python scripts/make_rh56f1_tip_tables.py [~/rl_ws/urdf/vendor/RH56F1]

Writes configs/hands/rh56f1_tips_<side>.npz from the vendor URDFs (RH56F1_R / RH56F1_L):
each finger's tip over its joint (0..1.5286 rad, 0.004 step), the thumb tip over thumb_1
(0.6..2.0944, 0.01) x thumb_2 (0..0.4746, 0.005), and the knuckle (finger joint) origins.
Metres, hand base frame. Re-run only when the hand model changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from motion_acq.hand.fk import TipFk
from motion_acq.hand.kinematic import FINGER_JOINT, FINGERS, table_path

URDF_NAME = {"index": "index", "middle": "middle", "ring": "ring", "pinky": "little"}


def main(argv: list[str]) -> int:
    root = Path(argv[1]).expanduser() if len(argv) > 1 else Path.home() / "rl_ws/urdf/vendor/RH56F1"
    finger_q = np.arange(0.0, 1.5285594 + 1e-9, 0.004)
    t1 = np.arange(0.6, 2.0943951 + 1e-9, 0.01)
    t2 = np.arange(0.0, 0.474555 + 1e-9, 0.005)
    for side, tag in (("right", "R"), ("left", "L")):
        fk = TipFk(root / f"RH56F1_{tag}" / "urdf" / f"RH56F1_{tag}.urdf")
        out = {"finger_q": finger_q}
        for f in FINGERS:
            out[f"tip_{f}"] = np.array([fk.link_position({FINGER_JOINT[f]: q}, f"{URDF_NAME[f]}_tip") for q in finger_q])
            out[f"knuckle_{f}"] = fk.link_position({}, f"{fk.prefix}{URDF_NAME[f]}_1")
        grid = np.array([(a, b) for a in t1 for b in t2])
        out["thumb_q"] = grid
        out["thumb_tips"] = np.array([fk.link_position({"thumb_1": a, "thumb_2": b}, "thumb_tip") for a, b in grid])
        path = table_path(side)
        np.savez_compressed(path, **out)
        print(f"{path}: fingers {len(finger_q)}, thumb {len(grid)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
