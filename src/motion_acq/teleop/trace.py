"""Per-frame teleop trace for arm-tuning analysis (10.11 user: the arms do not move as wanted).

``teleop-real --joint-debug`` writes one JSON line per published IK frame:
the tracker pose of every enabled side, the headset pose, the robot TCP
target, the IK / filtered joints, and the latest measured joints. The first
line holds the mapping setup (translation scale, controller->TCP calibration,
home TCP), and every arm start adds an ``anchor`` line with the headset
heading it took as robot forward. Offline, the operator's hand path and the
robot's commanded and measured path can then be compared frame by frame.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np


def _r(values: Any, digits: int = 5) -> list[float] | None:
    if values is None:
        return None
    return [
        round(float(v), digits) for v in np.asarray(values, dtype=np.float64).ravel()
    ]


class TeleopTrace:
    """Append-only JSON-lines trace; one file per teleop run."""

    def __init__(self, path: Path, arm_indices: Mapping[str, Sequence[int]]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.arm_indices = {
            side: [int(i) for i in idx] for side, idx in arm_indices.items()
        }
        self._file: IO[str] | None = self.path.open("a", encoding="utf-8", buffering=1)

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        record.setdefault("t", round(time.monotonic(), 4))
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _arm(self, q: np.ndarray | None) -> dict[str, list[float] | None]:
        if q is None:
            return {side: None for side in self.arm_indices}
        arr = np.asarray(q, dtype=np.float64)
        return {side: _r(arr[idx]) for side, idx in self.arm_indices.items()}

    def setup(self, **info: Any) -> None:
        self._write({"kind": "setup", **info})

    def anchor(
        self, side: str, *, heading_rad: float, source_pose7: Any, robot_pose7: Any
    ) -> None:
        self._write(
            {
                "kind": "anchor",
                "side": side,
                "heading_deg": None
                if not np.isfinite(heading_rad)
                else round(float(np.degrees(heading_rad)), 2),
                "source": _r(source_pose7),
                "robot": _r(robot_pose7),
            }
        )

    def event(self, kind: str, **info: Any) -> None:
        self._write({"kind": kind, **info})

    def frame(
        self,
        *,
        sample_time_ns: int,
        active: Sequence[str],
        tracked: Mapping[str, bool],
        source: Mapping[str, Any],
        hmd: Any,
        target: Mapping[str, Any],
        ik_q: np.ndarray,
        filtered_q: np.ndarray,
        feedback_q: np.ndarray | None,
        openings: Mapping[str, float],
    ) -> None:
        sides = list(self.arm_indices)
        self._write(
            {
                "kind": "frame",
                "ts": int(sample_time_ns),
                "active": list(active),
                "tracked": {s: bool(tracked.get(s, False)) for s in sides},
                "src": {s: _r(source.get(s)) for s in sides},
                "hmd": _r(hmd),
                "tgt": {s: _r(target.get(s)) for s in sides},
                "ik": self._arm(ik_q),
                "flt": self._arm(filtered_q),
                "fb": self._arm(feedback_q),
                "open": {s: round(float(openings.get(s, 0.0)), 3) for s in sides},
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
