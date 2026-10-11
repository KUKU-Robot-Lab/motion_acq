"""Flexion retargeting: each RH56F1 finger shows how far the operator's finger is bent.

10.11 user: a fingertip cannot land where the operator's does (the hands differ), the point was to
show each finger's bend, every finger like a one-joint gripper. The fingertip match (kinematic.py)
did not: the operator's long middle finger curled to a fist kept its tip outside the robot finger's
reach, the nearest robot tip was a nearly straight finger (10.11 logs: middle_1 at most 0.28 of
1.53 rad right, 0.46 left, with the glove at a full fist), and thumb_2 sat at 0.08-0.29 rad pulled
along by thumb_1.

Per robot joint, from the glove's own joint angles (SGCore hand model, rad):
* index_1 / middle_1 / ring_1: that finger's mcp + pip;
* pinky_1: the ring finger's (Nova 2 has no little-finger sensor: pinky_* equals ring_*);
* thumb_2: thumb_pip.
closure = (x - open - deadband) / (closed - open - deadband), clipped to 0..1, times the joint's
closed angle (deadband: a relaxed open hand reads a little above the most open one).

`closed` is the calibration's fist (the most bent of the fist and that finger's own example pose,
raw glove values): 10.11 logs, both hands, the fist read 2.5-3.0 rad mcp+pip in the calibration
and at the top of every session alike. `open` is not stable (10.11: the open hand read 0.3-0.8 in
a session, 1.4-2.0 in the right hand's 10.05 calibration and in the [켜기] reference takes), so it
starts at the calibration's open/flat and follows the operator's smoothed reading down: the
first open hand held for a moment sets it (smoothed: a brief overextension does not). From there it
only widens (never closes back up), and stays within max_span of `closed` so a glitch cannot open it
without bound. The [켜기] reference take (the operator holds the robot's home: fingers straight) sets
`open` to its reading when that is within reference_tol of the calibration's (a glove reading
shifted since the calibration, e.g. a SenseCom restart); a reading further up is a bent hand, not a
shift (10.11 takes: 0.6-1.3 rad above the open hand of the same session, which read like the
calibration), and leaves `open` as it is.

With [손가락별 맞춤] sweeps in the calibration (sweeps.py, 10.11) the reading of each joint is its own
glove value minus the other fingers' crosstalk, and `open` / `closed` start from the sweeps instead of
the example poses; the rest is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Mapping, Sequence

from motion_acq.hand.sweeps import INPUTS as SWEEP_INPUTS, SOURCES, FingerSweeps, source_values

# robot joint -> glove inputs summed for it
FLEX_INPUTS: dict[str, tuple[str, ...]] = {
    "index_1": ("index_mcp", "index_pip"),
    "middle_1": ("middle_mcp", "middle_pip"),
    "ring_1": ("ring_mcp", "ring_pip"),
    "pinky_1": ("ring_mcp", "ring_pip"),
    "thumb_2": ("thumb_pip",),
}
# example poses whose reading is "bent all the way" for that joint (the most bent one counts)
CLOSED_POSES: dict[str, tuple[str, ...]] = {
    "index_1": ("fist", "index"),
    "middle_1": ("fist", "middle"),
    "ring_1": ("fist", "ring"),
    "pinky_1": ("fist", "ring"),
    "thumb_2": ("fist", "thumb_bend", "thumb_opposed"),
}
OPEN_POSES = ("open", "flat")


@dataclass(frozen=True)
class FlexionConfig:
    # smallest closed - open span (rad of glove reading): a calibration whose open hand was half bent
    # still needs this much travel for a full curl
    finger_min_span: float = 1.2
    thumb_min_span: float = 0.25
    # `open` never goes further than this below `closed` (fingers: the glove reads mcp+pip 0..~3)
    finger_max_span: float = 3.2
    thumb_max_span: float = 1.0
    # per glove reading: the smoothed value that may lower `open` (0.02 at 120 Hz ticks ~ 0.4 s)
    open_alpha: float = 0.02
    # a reference take up to this far above the calibration's open hand is a shifted glove (rad)
    finger_reference_tol: float = 0.6
    thumb_reference_tol: float = 0.15
    # glove travel past `open` still shown as a straight finger (rad)
    finger_deadband: float = 0.15
    thumb_deadband: float = 0.03

    def __post_init__(self) -> None:
        if not (0 < self.finger_min_span <= self.finger_max_span and 0 < self.thumb_min_span <= self.thumb_max_span):
            raise ValueError("flexion: need 0 < min_span <= max_span")
        if not (0 <= self.finger_deadband < self.finger_min_span and 0 <= self.thumb_deadband < self.thumb_min_span):
            raise ValueError("flexion: need 0 <= deadband < min_span")
        if not 0 < self.open_alpha <= 1:
            raise ValueError("flexion: open_alpha must be in (0, 1]")


def flexion_config(raw: Mapping | None) -> FlexionConfig:
    raw = dict(raw or {})
    unknown = set(raw) - set(FlexionConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"flexion: unknown keys {sorted(unknown)}")
    return FlexionConfig(**{k: float(v) for k, v in raw.items()})


def reading(joint: str, signals: Mapping[str, float]) -> float | None:
    names = FLEX_INPUTS[joint]
    if not all(n in signals for n in names):
        return None
    return float(sum(float(signals[n]) for n in names))


class FlexionRetargeter:
    """Glove joint angles -> the bend of the RH56F1 fingers and thumb_2 (rad)."""

    def __init__(self, medians: Mapping[str, Mapping[str, float]], closed_rad: Mapping[str, float],
                 config: FlexionConfig = FlexionConfig(), sweeps: FingerSweeps | None = None) -> None:
        self.config = config
        self.sweeps = sweeps
        self.closed_rad = {j: float(closed_rad[j]) for j in FLEX_INPUTS}
        self.closed: dict[str, float] = {}
        self._open0: dict[str, float] = {}
        self.rest: dict[str, tuple[float, ...]] = {}
        if sweeps is not None:
            self.closed = {j: float(sweeps.closed[j]) for j in FLEX_INPUTS}
            self._open0 = {j: float(sweeps.open[j]) for j in FLEX_INPUTS}
        for j in FLEX_INPUTS if sweeps is None else ():
            bent = [v for p in CLOSED_POSES[j] if (v := reading(j, medians.get(p, {}))) is not None]
            straight = [v for p in OPEN_POSES if (v := reading(j, medians.get(p, {}))) is not None]
            if not bent or not straight:
                raise ValueError(f"flexion: the calibration has no {CLOSED_POSES[j]} / {OPEN_POSES} "
                                 f"readings of {FLEX_INPUTS[j]}")
            self.closed[j] = max(bent)
            self._open0[j] = min(straight)
        self.open: dict[str, float] = {}
        self._smooth: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Back to the calibration's open hand (a new session)."""
        self.open = {j: max(v, self.closed[j] - self._max_span(j)) for j, v in self._open0.items()}
        self.rest = dict(self.sweeps.rest) if self.sweeps is not None else {}
        if self.sweeps is not None:
            self.closed = {j: float(self.sweeps.closed[j]) for j in FLEX_INPUTS}
        self._smooth = {}

    def reading(self, joint: str, signals: Mapping[str, float]) -> float | None:
        """The joint's glove reading (with sweeps: minus the other fingers' crosstalk)."""
        if self.sweeps is None:
            return reading(joint, signals)
        return self.sweeps.corrected(joint, signals, self.rest)

    def _max_span(self, j: str) -> float:
        c = self.config
        return c.thumb_max_span if j == "thumb_2" else c.finger_max_span

    def _min_span(self, j: str) -> float:
        c = self.config
        return c.thumb_min_span if j == "thumb_2" else c.finger_min_span

    def take_reference(self, samples: Sequence[Mapping[str, float]]) -> list[str]:
        """The operator's straight hand (the [켜기] reference take): the joints whose `open` it set."""
        c, taken = self.config, []
        if self.sweeps is not None:
            self._take_rest(samples)
        for j in FLEX_INPUTS:
            values = [v for s in samples if (v := self.reading(j, s)) is not None]
            if not values:
                continue
            r = statistics.median(values)
            tol = c.thumb_reference_tol if j == "thumb_2" else c.finger_reference_tol
            if r <= self._open0[j] + tol:
                self.open[j] = max(r, self.closed[j] - self._max_span(j))
                taken.append(j)
        self._smooth = {}
        return taken

    def _take_rest(self, samples: Sequence[Mapping[str, float]]) -> None:
        """The reference hand is the crosstalk baseline: the thumb's always (its leak is linear, the open
        end absorbs the shift), a finger's when it reads like the sweeps' open hand (a bent finger is not)."""
        assert self.sweeps is not None
        for src in SOURCES:
            values = [v for s in samples if (v := source_values(src, s)) is not None]
            if not values:
                continue
            now = tuple(statistics.median(v[i] for v in values) for i in range(len(values[0])))
            if src == "thumb":
                # the thumb's leak is linear: a new baseline shifts the leaking joints' readings, both ends
                for j in FLEX_INPUTS:
                    w = self.sweeps.leak.get(j, {}).get("thumb")
                    if w is not None:
                        self.closed[j] += float(sum(k * (a - b) for k, a, b in zip(w, now, self.rest["thumb"])))
                self.rest[src] = now
            elif now[0] <= self.sweeps.rest[src][0] + self.config.finger_reference_tol:
                self.rest[src] = now

    def ready(self, signals: Mapping[str, float]) -> bool:
        if self.sweeps is not None:
            return all(n in signals for n in SWEEP_INPUTS)
        return all(reading(j, signals) is not None for j in FLEX_INPUTS)

    def closure(self, joint: str, x: float) -> float:
        c = self.config
        dead = c.thumb_deadband if joint == "thumb_2" else c.finger_deadband
        span = max(self.closed[joint] - self.open[joint], self._min_span(joint)) - dead
        return min(max((x - self.open[joint] - dead) / span, 0.0), 1.0)

    def solve(self, signals: Mapping[str, float]) -> dict[str, float]:
        """{joint: rad} for the four fingers and thumb_2; updates the open hand reading."""
        out = {}
        a = self.config.open_alpha
        for j in FLEX_INPUTS:
            x = self.reading(j, signals)
            if x is None:
                raise KeyError(f"flexion: no {FLEX_INPUTS[j]} in the glove signals (check ready())")
            prev = self._smooth.get(j)
            # the first reading of a session starts the smoothing no lower than `open`: a glitch there
            # must not open the range at once
            s = max(x, self.open[j]) if prev is None else prev + a * (x - prev)
            self._smooth[j] = s
            if s < self.open[j]:
                self.open[j] = max(s, self.closed[j] - self._max_span(j))
            out[j] = self.closure(j, x) * self.closed_rad[j]
        return out

    def ranges(self) -> dict[str, float]:
        """Log fields: flex_open_<joint>, flex_closed_<joint>."""
        out = {f"flex_open_{j}": v for j, v in self.open.items()}
        out.update({f"flex_closed_{j}": v for j, v in self.closed.items()})
        out["flex_sweeps"] = 1.0 if self.sweeps is not None else 0.0
        return out
