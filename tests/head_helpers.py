"""Shared helpers for the head tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from motion_acq.head.retarget import AxisConfig, RetargetConfig


def pose_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """HandUMI pose7 for an HMD turned yaw (left +) and pitched (up +)."""
    y, p = math.radians(yaw_deg) / 2, math.radians(-pitch_deg) / 2
    qz = np.array([0, 0, math.sin(y), math.cos(y)])
    qy = np.array([0, math.sin(p), 0, math.cos(p)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])
    return np.concatenate([[0.0, 0.0, 1.5], q])


def make_config(**overrides: object) -> RetargetConfig:
    base = RetargetConfig(
        pan=AxisConfig(home_deg=-2.9, range_deg=20.0),
        tilt=AxisConfig(home_deg=71.8, range_deg=15.0),
        deadband_deg=0.0,
        min_cutoff_hz=1e7,  # effectively unfiltered for geometry tests
        beta=0.0,
        max_velocity_deg_s=1e6,
        max_acceleration_deg_s2=1e9,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]
