"""Per-finger sweeps: each robot finger 0..1 from that operator finger alone, glove crosstalk removed.

10.11 user: with the flexion map, moving one finger still moved others on the robot (the thumb tip alone
moved the index); each finger should map 0..1 from moving that fingertip alone. The glove readings are
coupled: in the 10.11 follow logs, windows where the glove thumb moved and the middle / ring stayed had
the index reading move > 0.2 rad in 14-30 % of them (left hand every session, right in two of three),
and moving the middle alone moved the index reading 0.2-1.1 x as much. A per-finger 0..1 range alone
keeps that: the index reading itself moves with the thumb. So the operator moves one finger at a time
(open and still for a moment, then bend and straighten it a few times), and from those recordings:

* each finger's own range: open = its reading with the hand still and open, closed = its bent end
  (95th percentile) in its own sweep and in the fist sweep, the lower of the two so both reach 1;
* the crosstalk: in finger k's sweep, how much finger j's reading moves with k (least squares through
  the open hand, kept only where it explains the movement).

At run time finger j reads its own value minus what the other fingers leak into it:
    r_j' = r_j - sum_k leak[j][k] . d_k
with d_k = max(s_k - rest_k, 0) for a finger (its mcp+pip bend: crosstalk comes from bending) and the
thumb's (thumb_pip, thumb_brake, thumb_mcp) - rest for the thumb (it bends and swings across the palm,
both can leak). A finger that moves alone then moves only its robot finger. The pinky follows the ring
(Nova 2 has no little-finger sensor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics
from typing import Callable, Mapping, Sequence

import numpy as np

# sweep -> operator prompt
SWEEPS: dict[str, str] = {
    "thumb": "엄지만: 네 손가락은 편 채 가만히, 엄지 끝을 굽혔다 펴고, 손바닥을 가로질러 새끼 뿌리 쪽으로 갔다 오기",
    "index": "검지만: 다른 손가락은 편 채 가만히, 검지만 끝까지 굽혔다 펴기",
    "middle": "중지만: 다른 손가락은 편 채 가만히, 중지만 끝까지 굽혔다 펴기",
    "ring": "약지·새끼만: 검지·중지·엄지는 편 채, 약지와 새끼를 끝까지 굽혔다 펴기",
    "fist": "주먹: 다섯 손가락을 같이 끝까지 쥐었다 펴기",
}
SOURCES = ("thumb", "index", "middle", "ring")  # sweeps whose movement can leak into the other fingers
THUMB_INPUTS = ("thumb_pip", "thumb_brake", "thumb_mcp")
# robot joint -> glove inputs summed for its reading (as flexion.FLEX_INPUTS)
JOINT_INPUTS: dict[str, tuple[str, ...]] = {
    "index_1": ("index_mcp", "index_pip"),
    "middle_1": ("middle_mcp", "middle_pip"),
    "ring_1": ("ring_mcp", "ring_pip"),
    "pinky_1": ("ring_mcp", "ring_pip"),
    "thumb_2": ("thumb_pip",),
}
OWN_SOURCE = {"thumb_2": "thumb", "index_1": "index", "middle_1": "middle", "ring_1": "ring", "pinky_1": "ring"}
INPUTS = tuple(dict.fromkeys(THUMB_INPUTS + tuple(n for ins in JOINT_INPUTS.values() for n in ins)))
JOINT_KO = {"thumb_2": "엄지", "index_1": "검지", "middle_1": "중지", "ring_1": "약지", "pinky_1": "새끼"}
SOURCE_KO = {"thumb": "엄지", "index": "검지", "middle": "중지", "ring": "약지·새끼", "fist": "주먹"}

MIN_MOVE = {"thumb": 0.15, "index": 0.5, "middle": 0.5, "ring": 0.5, "fist": 0.5}  # rad a sweep must travel
REST_MAX_STD = 0.05  # rad: the open hand at the start of a sweep is held still
LEAK_MIN_R2 = 0.3  # a leak must explain this much of the other finger's movement
LEAK_MIN_EFFECT = 0.1  # rad of the other finger's reading (95th percentile over the sweep)
LEAK_MAX = 0.8  # |coefficient| cap: near 1 is a finger the operator cannot move alone, not crosstalk
RIDGE = 0.01  # thumb leak fit: ridge weight relative to the mean input variance


class SweepError(ValueError):
    pass


def joint_reading(joint: str, signals: Mapping[str, float]) -> float | None:
    names = JOINT_INPUTS[joint]
    if not all(n in signals for n in names):
        return None
    return float(sum(float(signals[n]) for n in names))


def source_values(name: str, signals: Mapping[str, float]) -> tuple[float, ...] | None:
    """The glove values of a source finger: the thumb's three, a finger's mcp+pip."""
    if name == "thumb":
        if not all(n in signals for n in THUMB_INPUTS):
            return None
        return tuple(float(signals[n]) for n in THUMB_INPUTS)
    x = joint_reading(f"{name}_1", signals)
    return None if x is None else (x,)


def source_delta(name: str, values: Sequence[float], rest: Sequence[float]) -> np.ndarray:
    d = np.asarray(values, dtype=float) - np.asarray(rest, dtype=float)
    return d if name == "thumb" else np.maximum(d, 0.0)


@dataclass(frozen=True)
class FingerSweeps:
    rest: Mapping[str, tuple[float, ...]]  # source -> its glove values with the hand open
    leak: Mapping[str, Mapping[str, tuple[float, ...]]]  # robot joint -> source -> coefficients
    open: Mapping[str, float]  # robot joint -> corrected reading of the open hand
    closed: Mapping[str, float]  # robot joint -> corrected reading of the finger bent all the way
    recorded: str = ""
    travel: Mapping[str, float] = field(default_factory=dict)  # sweep -> how far it moved (report)

    def corrected(self, joint: str, signals: Mapping[str, float],
                  rest: Mapping[str, Sequence[float]] | None = None) -> float | None:
        """The joint's reading minus the other fingers' leak (rest: the sources' open values in use)."""
        x = joint_reading(joint, signals)
        if x is None:
            return None
        rest = self.rest if rest is None else rest
        for src, w in self.leak.get(joint, {}).items():
            v = source_values(src, signals)
            if v is None:
                return None
            x -= float(np.dot(w, source_delta(src, v, rest[src])))
        return x

    def to_dict(self) -> dict:
        return {"rest": {k: [float(x) for x in v] for k, v in self.rest.items()},
                "leak": {j: {s: [float(x) for x in w] for s, w in m.items()} for j, m in self.leak.items()},
                "open": {k: float(v) for k, v in self.open.items()},
                "closed": {k: float(v) for k, v in self.closed.items()},
                "travel": {k: float(v) for k, v in self.travel.items()},
                "recorded": self.recorded}

    @classmethod
    def from_dict(cls, raw: Mapping) -> FingerSweeps:
        try:
            out = cls({str(k): tuple(float(x) for x in v) for k, v in raw["rest"].items()},
                      {str(j): {str(s): tuple(float(x) for x in w) for s, w in (m or {}).items()}
                       for j, m in raw["leak"].items()},
                      {str(k): float(v) for k, v in raw["open"].items()},
                      {str(k): float(v) for k, v in raw["closed"].items()},
                      str(raw.get("recorded", "")),
                      {str(k): float(v) for k, v in (raw.get("travel") or {}).items()})
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise SweepError(f"finger_sweeps: {exc!r}") from exc
        sizes_ok = all(len(out.rest.get(s, ())) == (3 if s == "thumb" else 1) for s in SOURCES) and all(
            len(w) == len(out.rest[s]) for m in out.leak.values() for s, w in m.items() if s in out.rest)
        if not sizes_ok or set(out.open) != set(JOINT_INPUTS) or set(out.closed) != set(JOINT_INPUTS) \
                or any(s not in SOURCES for m in out.leak.values() for s in m):
            raise SweepError("finger_sweeps: incomplete (record [손가락별 맞춤] again)")
        return out


Recording = tuple[Sequence[Mapping[str, float]], Sequence[Mapping[str, float]]]  # (open and still, moving)


def unsteady(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """{input: spread} of the inputs that moved while the open hand was held (empty: still)."""
    out = {}
    for n in INPUTS:
        values = [float(s[n]) for s in samples if n in s]
        if len(values) >= 2 and statistics.pstdev(values) > REST_MAX_STD:
            out[n] = statistics.pstdev(values)
    return out


def _need(value):
    if value is None:  # fit_sweeps checks first that every sample has every input
        raise SweepError("장갑 값이 빠진 샘플")
    return value


def _rest_of(name: str, samples: Sequence[Mapping[str, float]]) -> tuple[float, ...]:
    values = np.array([_need(source_values(name, s)) for s in samples])
    return tuple(float(v) for v in np.median(values, axis=0))


def travel(name: str, rec: Recording) -> float:
    """95th percentile of how far the sweep moved from its open hand (fist: the middle finger)."""
    rest, moving = rec
    src = "middle" if name == "fist" else name
    r0 = _rest_of(src, rest)
    d = [float(np.linalg.norm(source_delta(src, _need(source_values(src, s)), r0))) for s in moving]
    return float(np.percentile(d, 95))


def _fit_leak(src: str, joint: str, rec: Recording) -> tuple[float, ...] | None:
    """Coefficients of joint's reading on the source's movement in the source's sweep (None: no leak)."""
    rest, moving = rec
    s0 = _rest_of(src, rest)
    D = np.array([source_delta(src, _need(source_values(src, s)), s0) for s in moving])
    r0 = statistics.median(_need(joint_reading(joint, s)) for s in rest)
    dr = np.array([_need(joint_reading(joint, s)) - r0 for s in moving])
    gram = D.T @ D
    lam = RIDGE * float(np.trace(gram)) / D.shape[1] if src == "thumb" else 0.0
    w = np.linalg.solve(gram + (lam + 1e-9) * np.eye(D.shape[1]), D.T @ dr)
    w = np.clip(w, -LEAK_MAX, LEAK_MAX)
    pred = D @ w
    r2 = 1.0 - float((dr - pred) @ (dr - pred)) / max(float(dr @ dr), 1e-9)
    if r2 < LEAK_MIN_R2 or float(np.percentile(np.abs(pred), 95)) < LEAK_MIN_EFFECT:
        return None
    return tuple(float(v) for v in w)


def fit_sweeps(recordings: Mapping[str, Recording], recorded: str = "") -> FingerSweeps:
    """Ranges and crosstalk from the five sweeps (SWEEPS: thumb, index, middle, ring, fist)."""
    missing = [k for k in SWEEPS if k not in recordings]
    if missing:
        raise SweepError(f"기록 없음: {missing}")
    for k, (rest, moving) in recordings.items():
        if len(rest) < 5 or len(moving) < 20:
            raise SweepError(f"{SOURCE_KO.get(k, k)}: 샘플이 너무 적다 (편 손 {len(rest)}, 움직임 {len(moving)})")
        bad = [s for s in list(rest) + list(moving) if not all(n in s for n in INPUTS)]
        if bad:
            raise SweepError(f"{SOURCE_KO.get(k, k)}: 장갑 값이 빠졌다 ({sorted(set(INPUTS) - set(bad[0]))})")
    moved = {k: travel(k, recordings[k]) for k in SWEEPS}
    short = [k for k in SWEEPS if moved[k] < MIN_MOVE[k]]
    if short:
        raise SweepError("거의 움직이지 않은 기록: " + ", ".join(
            f"{SOURCE_KO[k]} {moved[k]:.2f} < {MIN_MOVE[k]}" for k in short))
    all_rest = [s for rest, _ in recordings.values() for s in rest]
    rest_value = {k: _rest_of(k, all_rest) for k in SOURCES}
    leak: dict[str, dict[str, tuple[float, ...]]] = {j: {} for j in JOINT_INPUTS}
    for src in SOURCES:
        for joint in JOINT_INPUTS:
            if OWN_SOURCE[joint] != src and (w := _fit_leak(src, joint, recordings[src])) is not None:
                leak[joint][src] = w
    partial = FingerSweeps(rest_value, leak, {}, {}, recorded, moved)
    corrected: Callable[[str, Mapping[str, float]], float] = lambda j, s: _need(partial.corrected(j, s))  # noqa: E731
    open_, closed = {}, {}
    for joint in JOINT_INPUTS:
        open_[joint] = statistics.median(corrected(joint, s) for s in all_rest)
        own = float(np.percentile([corrected(joint, s) for s in recordings[OWN_SOURCE[joint]][1]], 95))
        fist = float(np.percentile([corrected(joint, s) for s in recordings["fist"][1]], 95))
        closed[joint] = min(own, fist) if min(own, fist) > open_[joint] else max(own, fist)
    return FingerSweeps(rest_value, leak, open_, closed, recorded, moved)


def report(sweeps: FingerSweeps) -> list[str]:
    """Operator-facing lines: each finger's range and the crosstalk removed."""
    lines = ["  손가락별 범위 (장갑 값, 편 손 → 끝까지 굽힘, 간섭 뺀 값):"]
    lines += [f"    {JOINT_KO[j]}: {sweeps.open[j]:.2f} → {sweeps.closed[j]:.2f}"
              for j in JOINT_INPUTS if j != "pinky_1"]
    leaks = [(j, s, w) for j, m in sweeps.leak.items() if j != "pinky_1" for s, w in m.items()]
    if not leaks:
        return lines + ["  손가락 사이 간섭: 없음"]
    lines.append("  손가락 사이 간섭 (빼고 따라간다):")
    for j, s, w in leaks:
        coef = " ".join(f"{n.split('_')[1]} {v:+.2f}" for n, v in zip(THUMB_INPUTS, w)) if s == "thumb" \
            else f"{w[0]:+.2f} 배"
        lines.append(f"    {SOURCE_KO[s]} → {JOINT_KO[j]}: {coef}")
    return lines
