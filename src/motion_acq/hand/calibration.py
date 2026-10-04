"""Per-user, per-side Nova 2 calibration: feature value at the open/closed poses."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
            raise PoseSpanError(feature, (open_pose, closed_pose), lo, hi, min_span)
        ranges[feature] = FeatureRange(lo, hi)
    return HandCalibration(side, user, ranges)


class PoseSpanError(CalibrationError):
    """Two poses that should differ gave (almost) the same feature value."""

    def __init__(self, feature: str, poses: tuple[str, str], lo: float, hi: float, min_span: float) -> None:
        self.feature, self.poses, self.lo, self.hi = feature, poses, lo, hi
        super().__init__(f"{feature}: {poses[0]} {lo:.3f} vs {poses[1]} {hi:.3f} rad differ by "
                         f"less than {min_span} rad; redo those poses")


# -- operator session (the ROS calibrate node supplies ask / record / say) -----------------------------

MAX_POSE_STD_RAD = 0.05  # a calibrated feature moving more than this while "still" -> ask again
SIDE_KO = {"right": "오른손", "left": "왼손"}
FEATURE_KO = {"index": "검지", "middle": "중지", "ring": "약지", "pinky": "새끼",
              "thumb_bend": "엄지 굽힘", "thumb_opposition": "엄지 맞대기"}


def pose_features(feature_poses: Mapping[str, tuple[str, str]], pose: str) -> tuple[str, ...]:
    """The features whose open or closed end this pose records."""
    return tuple(f for f, ends in feature_poses.items() if pose in ends)


def unsteady_features(
    samples: Sequence[Mapping[str, float]], names: Sequence[str], max_std: float,
) -> dict[str, float]:
    """{feature: std} for the given features that moved more than max_std."""
    spreads = {n: statistics.pstdev(s[n] for s in samples) for n in names}
    return {n: v for n, v in spreads.items() if v > max_std}


def run_session(
    *,
    side: str,
    user: str,
    poses: Mapping[str, str],
    feature_poses: Mapping[str, tuple[str, str]],
    min_span: float,
    ask: Callable[[str], None],
    record: Callable[[str], Sequence[Mapping[str, float]]],
    say: Callable[[str], None],
    max_tries: int = 3,
    max_std: float = MAX_POSE_STD_RAD,
) -> HandCalibration:
    """Walk the operator through every pose; a bad take repeats that pose only.

    ``ask`` shows the prompt and returns once the operator is in the pose (Enter),
    ``record`` returns that pose's feature samples. A pose whose calibrated
    features moved is asked again; two poses that came out the same are both
    asked again. Gives up (CalibrationError) after ``max_tries`` of either.
    """
    order = list(poses)
    side_ko = SIDE_KO.get(side, side)

    def take(pose: str) -> Sequence[Mapping[str, float]]:
        step = f"[{side_ko} {order.index(pose) + 1}/{len(order)}]"
        for attempt in range(1, max_tries + 1):
            ask(f"{step} {poses[pose]}. 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
            say("  기록 중: 그대로 멈춰 있으세요")
            samples = record(pose)
            moved = unsteady_features(samples, pose_features(feature_poses, pose), max_std)
            if not moved:
                say(f"  기록됨 ({len(samples)} 샘플)")
                return samples
            names = ", ".join(f"{FEATURE_KO.get(n, n)} {v:.2f} rad" for n, v in moved.items())
            say(f"  움직였습니다 ({names}): 같은 자세를 다시 합니다 ({attempt}/{max_tries})")
        raise CalibrationError(f"{step} 자세를 {max_tries}번 모두 움직였습니다")

    samples = {pose: take(pose) for pose in order}
    for attempt in range(1, max_tries + 1):
        try:
            return calibrate(side=side, user=user, pose_samples=samples,
                             feature_poses=feature_poses, min_span=min_span)
        except PoseSpanError as exc:
            if attempt == max_tries:
                raise CalibrationError(f"{max_tries}번 해도 {exc}") from exc
            redo = [p for p in order if p in exc.poses]
            say(f"  {FEATURE_KO.get(exc.feature, exc.feature)} 값이 두 자세에서 거의 같습니다 "
                f"({exc.lo:.2f} vs {exc.hi:.2f} rad, {min_span} 이상 달라야 함): "
                f"{', '.join(f'{order.index(p) + 1}번' for p in redo)} 자세를 다시 합니다")
            for pose in redo:
                samples[pose] = take(pose)
    raise AssertionError("unreachable")
