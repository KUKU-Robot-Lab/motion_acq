"""Per-user, per-side Nova 2 calibration: feature value at the open/closed poses."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import yaml

SCHEMA = "motion_acq/hand_calibration/v1"
MIN_LOADED_SPAN_RAD = 1e-3  # a hand-edited file must still have a real range


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureRange:
    open: float
    closed: float

    def normalize(self, value: float) -> float:
        n = (float(value) - self.open) / (self.closed - self.open)
        return min(max(n, 0.0), 1.0)


@dataclass(frozen=True)
class HandCalibration:
    side: str
    user: str
    ranges: Mapping[str, FeatureRange]
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    glove: str = "nova2"

    def normalize(self, feats: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in self.ranges if name not in feats]
        if missing:
            raise CalibrationError(f"features missing from the sample: {missing}")
        return {name: r.normalize(feats[name]) for name, r in self.ranges.items()}

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "glove": self.glove, "side": self.side, "user": self.user,
            "created": self.created,
            "ranges": {k: {"open": r.open, "closed": r.closed} for k, r in self.ranges.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, side: str | None = None) -> HandCalibration:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema") != SCHEMA:
            raise CalibrationError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
        if side is not None and raw.get("side") != side:
            raise CalibrationError(f"{path} is a {raw.get('side')} hand calibration, not {side}")
        ranges = {}
        for k, v in (raw.get("ranges") or {}).items():
            lo, hi = float(v["open"]), float(v["closed"])
            if not (math.isfinite(lo) and math.isfinite(hi)) or abs(hi - lo) < MIN_LOADED_SPAN_RAD:
                raise CalibrationError(f"{path}: {k} open {lo} / closed {hi} is not a usable range")
            ranges[k] = FeatureRange(lo, hi)
        if not ranges:
            raise CalibrationError(f"{path}: no feature ranges")
        return cls(raw["side"], str(raw.get("user", "")), ranges, str(raw.get("created", "")))


def calibrate(
    *,
    side: str,
    user: str,
    pose_samples: Mapping[str, Sequence[Mapping[str, float]]],
    feature_poses: Mapping[str, tuple[str, str]],
    min_span: float,
) -> HandCalibration:
    """Median feature value per pose -> open/closed range per feature."""
    medians: dict[str, dict[str, float]] = {}
    for pose, samples in pose_samples.items():
        if not samples:
            raise CalibrationError(f"pose {pose!r} has no samples")
        names = samples[0].keys()
        medians[pose] = {n: statistics.median(s[n] for s in samples) for n in names}
    ranges = {}
    for feature, (open_pose, closed_pose) in feature_poses.items():
        for pose in (open_pose, closed_pose):
            if pose not in medians:
                raise CalibrationError(f"pose {pose!r} (for {feature}) was not recorded")
        lo, hi = medians[open_pose][feature], medians[closed_pose][feature]
        if abs(hi - lo) < min_span:
            raise CalibrationError(
                f"{feature}: {open_pose} {lo:.3f} vs {closed_pose} {hi:.3f} rad differ by "
                f"less than {min_span} rad; redo those poses"
            )
        ranges[feature] = FeatureRange(lo, hi)
    return HandCalibration(side, user, ranges)
