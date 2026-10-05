"""Per-user, per-side Nova 2 calibration: example poses -> glove-to-RH56F1 map (v2, 10.05).

The operator holds each example pose of configs/hands/nova2_to_rh56f1.yaml; the samples
and the robot targets fit motion_acq.hand.example_map. The file is kept per user
(configs/hands/calibration/<user>_<side>.yaml) and reused across sessions (10.05 user:
calibrating every time is not workable, that is what the user name is for). After a
SenseCom restart or a glove power cycle the raw readings can shift; rezero() takes the
open hand for two seconds and shifts the inputs, without redoing the examples.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from motion_acq.hand.example_map import ConflictError, ExampleMap, ExampleMapError, fit_groups, input_kind

SCHEMA = "motion_acq/hand_calibration/v3"  # v3: one map per joint group, thumb tip position input
OLD_SCHEMAS = ("motion_acq/hand_calibration/v1", "motion_acq/hand_calibration/v2")
OPEN_POSE = "open"


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class HandCalibration:
    side: str
    user: str
    models: Mapping[str, ExampleMap]  # joint group -> its map
    medians: Mapping[str, Mapping[str, float]]  # example pose -> input -> median (raw, at calibration)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    rezeroed: str | None = None
    glove: str = "nova2"

    def predict(self, signals: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for model in self.models.values():
            out.update(model.predict(signals))
        return out

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(j for m in self.models.values() for j in m.joints)

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(n for m in self.models.values() for n in m.inputs))

    def missing_inputs(self, signals: Mapping[str, float]) -> list[str]:
        return [n for n in self.inputs if n not in signals]

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "glove": self.glove, "side": self.side, "user": self.user,
            "created": self.created, "rezeroed": self.rezeroed,
            "medians": {p: {k: float(v) for k, v in m.items()} for p, m in self.medians.items()},
            "models": {g: m.to_dict() for g, m in self.models.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, *, side: str | None = None) -> HandCalibration:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if raw.get("schema") in OLD_SCHEMAS:
            raise CalibrationError(f"{path} is an older calibration format: calibrate again (example poses)")
        if raw.get("schema") != SCHEMA:
            raise CalibrationError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
        if side is not None and raw.get("side") != side:
            raise CalibrationError(f"{path} is a {raw.get('side')} hand calibration, not {side}")
        try:
            models = {str(g): ExampleMap.from_dict(m) for g, m in (raw.get("models") or {}).items()}
        except ExampleMapError as exc:
            raise CalibrationError(f"{path}: {exc}") from exc
        if not models:
            raise CalibrationError(f"{path}: no maps")
        medians = {str(p): {str(k): float(v) for k, v in m.items()} for p, m in (raw.get("medians") or {}).items()}
        if OPEN_POSE not in medians:
            raise CalibrationError(f"{path}: no {OPEN_POSE} pose medians")
        return cls(str(raw["side"]), str(raw.get("user", "")), models, medians, str(raw.get("created", "")),
                   raw.get("rezeroed"))


def rezero(calibration: HandCalibration, open_samples: Sequence[Mapping[str, float]]) -> HandCalibration:
    """Shift the inputs so today's open hand reads like the calibration's (2 s, no examples)."""
    now = {n: statistics.median(s[n] for s in open_samples) for n in calibration.inputs
           if open_samples and all(n in s for s in open_samples)}
    if not now:
        raise CalibrationError("no glove values in the open-hand recording")
    ref = calibration.medians[OPEN_POSE]
    models = {g: m.with_offset([now[n] - ref[n] if n in now and n in ref else 0.0 for n in m.inputs])
              for g, m in calibration.models.items()}
    return HandCalibration(calibration.side, calibration.user, models, calibration.medians,
                           calibration.created, time.strftime("%Y-%m-%dT%H:%M:%S"), calibration.glove)


# -- operator session (the ROS calibrate node supplies ask / record / say) -----------------------------

MAX_STD = {"angle": 0.08, "tipdist": 0.10, "position": 6.0}  # rad / fraction of the distance / mm: "held still"
SIDE_KO = {"right": "오른손", "left": "왼손"}


def unsteady_inputs(samples: Sequence[Mapping[str, float]], inputs: Sequence[str]) -> dict[str, float]:
    """{input: spread} of the inputs that moved while the pose was held."""
    out = {}
    for n in inputs:
        values = [s[n] for s in samples if n in s]
        if len(values) < 2:
            continue
        spread = statistics.pstdev(values)
        kind = input_kind(n)
        limit = MAX_STD["tipdist"] * abs(statistics.fmean(values)) if kind == "tipdist" else MAX_STD[kind]
        if spread > limit:
            out[n] = spread
    return out


def poses_to_redo(pairs: Sequence[tuple[str, str]]) -> list[str]:
    """Fewest poses covering every conflicting pair: one badly done pose usually collides with
    many others (an "index only" done flat collides with every flat-index pose); redo just it."""
    left, redo = [tuple(p) for p in pairs], []
    while left:
        counts: dict[str, int] = {}
        for pair in left:
            for pose in pair:
                counts[pose] = counts.get(pose, 0) + 1
        pick = max(counts, key=lambda pose: counts[pose])
        redo.append(pick)
        left = [pair for pair in left if pick not in pair]
    return redo


def run_session(
    *,
    side: str,
    user: str,
    groups: Mapping[str, tuple[Sequence[str], Sequence[str]]],  # joint group -> (inputs, joints)
    examples: Mapping[str, tuple[str, Mapping[str, float]]],  # pose -> (prompt, robot target)
    ask: Callable[[str], None],
    record: Callable[[str], Sequence[Mapping[str, float]]],
    say: Callable[[str], None],
    max_tries: int = 3,
) -> HandCalibration:
    """Walk the operator through every example pose; a bad take repeats that pose only.

    ``ask`` shows the prompt and returns once the operator is in the pose (Enter), ``record``
    returns the glove signals of that pose (CalibrationError: no usable recording). A pose
    that moved, or two poses the glove cannot tell apart, are asked again."""
    order = list(examples)
    inputs = list(dict.fromkeys(n for ins, _ in groups.values() for n in ins))
    if OPEN_POSE not in order:
        raise CalibrationError(f"the examples need an {OPEN_POSE!r} pose")
    side_ko = SIDE_KO.get(side, side)

    def take(pose: str) -> Sequence[Mapping[str, float]]:
        step = f"[{side_ko} {order.index(pose) + 1}/{len(order)}]"
        for attempt in range(1, max_tries + 1):
            ask(f"{step} {examples[pose][0]}. 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
            say("  기록 중: 그대로 멈춰 있으세요")
            try:
                samples = record(pose)
            except CalibrationError as exc:
                say(f"  기록 실패 ({exc}): 같은 자세를 다시 합니다 ({attempt}/{max_tries})")
                continue
            moved = unsteady_inputs(samples, inputs)
            if not moved:
                say(f"  기록됨 ({len(samples)} 샘플)")
                return samples
            names = ", ".join(f"{n} {v:.2f}" for n, v in moved.items())
            say(f"  움직였습니다 ({names}): 같은 자세를 다시 합니다 ({attempt}/{max_tries})")
        raise CalibrationError(f"{step} 자세를 {max_tries}번 해도 기록하지 못했습니다")

    samples = {pose: take(pose) for pose in order}
    targets = {p: examples[p][1] for p in order}
    redone = False
    while True:
        try:
            models = fit_groups(groups, samples, targets, OPEN_POSE, check_conflicts=not redone)
        except ConflictError as exc:
            # 10.05: a human cannot always isolate a finger (ring curls the middle too): ask once
            # for the poses that collide, then keep what the glove gives (the map averages them)
            redone = True
            redo = poses_to_redo([c.poses for c in exc.conflicts])
            say(f"  장갑에서 비슷하게 읽히는 자세가 있습니다 ({exc}): "
                f"{', '.join(f'{order.index(p) + 1}번' for p in redo)} 자세를 한 번 더 합니다 "
                "(그래도 비슷하면 그대로 저장합니다)")
            for pose in redo:
                samples[pose] = take(pose)
            continue
        except ExampleMapError as exc:
            raise CalibrationError(str(exc)) from exc
        if redone:
            try:
                fit_groups(groups, samples, targets, OPEN_POSE)
            except ConflictError as exc:
                say(f"  주의: 여전히 비슷하게 읽히는 자세가 있어 둘의 중간으로 따라갑니다 ({exc})")
        medians = {p: {n: statistics.median(s[n] for s in samples[p]) for n in inputs} for p in order}
        return HandCalibration(side, user, models, medians)


def fit_error(calibration: HandCalibration, examples: Mapping[str, tuple[str, Mapping[str, float]]]) -> float:
    """Largest joint error (rad) of the map at the example poses."""
    worst = 0.0
    for pose, (_, target) in examples.items():
        if pose in calibration.medians:
            q = calibration.predict(calibration.medians[pose])
            worst = max(worst, max(abs(q[j] - target[j]) for j in target))
    return worst
