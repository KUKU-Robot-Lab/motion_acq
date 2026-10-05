"""Shared hand test fixtures: a calibration fitted from the synthetic operator's examples."""

from __future__ import annotations

import numpy as np

from motion_acq.hand.calibration import HandCalibration, run_session
from motion_acq.hand.retarget import load_hand_retarget_config
from motion_acq.hand.synthetic import POSE_ANGLES

CONFIG = load_hand_retarget_config()


def held(pose: str, n: int = 30, seed: int = 0) -> list[dict[str, float]]:
    """A held example pose: the synthetic values with a little sensor noise."""
    rng = np.random.default_rng(seed + sorted(POSE_ANGLES).index(pose))
    out = []
    for _ in range(n):
        out.append({k: v + float(rng.normal(0.0, 0.5 if k.startswith("tipdist") else 0.004))
                    for k, v in POSE_ANGLES[pose].items()})
    return out


def examples() -> dict:
    return {p: (e.prompt, e.target) for p, e in CONFIG.examples.items()}


def make_calibration(side: str = "right") -> HandCalibration:
    return run_session(side=side, user="test", groups=CONFIG.groups, examples=examples(),
                       ask=lambda text: None, record=lambda pose: held(pose), say=lambda text: None)
