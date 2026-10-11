"""Last seconds of OpenArm commands, saved when the streamer stops on an error.

10.04 21:24:17 the right arm stopped 4 s after Space on "joint7 following error
0.351 rad" with no record of what it had been asked to do. The streamer keeps
the last TRACE_TICKS ticks (target from teleop, commanded after the speed limit,
measured) and writes them to logs/real/following_error_<side>_<time>.npz.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np

TRACE_DIR = Path(__file__).resolve().parents[4] / "logs" / "real"
TRACE_TICKS = 300  # 3 s at the 100 Hz command rate


class CommandTrace:
    """Ring buffer of (t, target, commanded, measured) per side."""

    def __init__(self, ticks: int = TRACE_TICKS) -> None:
        self._rows: deque[tuple[float, dict, dict, dict]] = deque(maxlen=int(ticks))

    def add(self, t: float, commanded: dict[str, np.ndarray], measured: dict[str, np.ndarray],
            target: dict[str, np.ndarray]) -> None:
        def copy(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {side: np.array(q, dtype=np.float32) for side, q in values.items()}

        self._rows.append((float(t), copy(commanded), copy(measured), copy(target)))

    def frozen_s(self, side: str, joint: int) -> float:
        """How long the newest measured value of one joint has stayed bit-identical (s).

        10.11: left joint7 read one value for 300 ticks in two faults while the other joints moved; a motor
        that stops answering leaves the SDK's last value in place."""
        rows = [r for r in self._rows if side in r[2]]
        if not rows:
            return 0.0
        last = rows[-1][2][side][joint]
        start = rows[-1][0]
        for t, _, measured, _ in reversed(rows):
            if measured[side][joint] != last:
                break
            start = t
        return rows[-1][0] - start

    def dump(self, path: Path) -> Path:
        rows = list(self._rows)
        arrays: dict[str, np.ndarray] = {"t": np.array([r[0] for r in rows], dtype=np.float64)}
        sides = rows[-1][1].keys() if rows else ()
        for side in sides:
            for index, name in ((1, "commanded"), (2, "measured"), (3, "target")):
                arrays[f"{side}_{name}"] = np.stack([r[index][side] for r in rows])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path


FROZEN_S = 0.2  # a joint at 100 Hz never reads the same float this long while it is being moved


def following_error_report(trace: CommandTrace, side: str, joint: int, commanded: float, measured: float,
                           limit: float, target: float, motor: str | None = None) -> str:
    """Error text with the angles, the motor's own state, and where the trace went (or why it did not)."""
    text = (f"OpenArm {side} joint{joint + 1} following error {abs(commanded - measured):.3f} rad exceeds "
            f"{limit:.3f} rad: commanded {math.degrees(commanded):.1f} deg, measured "
            f"{math.degrees(measured):.1f} deg, teleop target {math.degrees(target):.1f} deg.")
    frozen = trace.frozen_s(side, joint)
    if frozen >= FROZEN_S:
        text += (f" Its reading has not changed for {frozen:.2f} s: the motor stopped answering or stopped "
                 "moving (check its cable / connector and power; not a teleop target problem).")
    if motor:
        text += f" Motor: {motor}."
    path = TRACE_DIR / f"following_error_{side}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    try:
        return f"{text} Trace: {trace.dump(path)}"
    except OSError as exc:
        return f"{text} (trace not saved: {exc})"
