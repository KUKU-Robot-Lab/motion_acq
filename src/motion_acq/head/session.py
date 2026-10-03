"""Head teleop loop: tracking sample -> retargeter -> driver, with logging.

Independent of the arm pipeline (own process, own enable). Faults hold the
head; they never touch the arms.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

import numpy as np

from motion_acq.head.dynamixel import HeadAlertError, HeadBusError, HeadDriver
from motion_acq.head.retarget import HeadRetargeter, HeadState, HeadStep
from motion_acq.tracking.base import ControllerPairSample

log = logging.getLogger(__name__)

HARDWARE_CHECK_PERIOD_S = 1.0
COMMAND_EPSILON_DEG = 0.01


class HmdSource(Protocol):
    def latest(self) -> ControllerPairSample: ...


@dataclass
class SessionStats:
    cycles: int = 0
    commands: int = 0
    holds: int = 0
    faults: int = 0
    reanchors: int = 0


class HeadSession:
    """One anchored run. ``tick()`` is a single control cycle."""

    def __init__(
        self,
        tracker: HmdSource,
        driver: HeadDriver,
        retargeter: HeadRetargeter,
        *,
        auto_start_delay_s: float,
        max_consecutive_faults: int,
        log_file: IO[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tracker = tracker
        self.driver = driver
        self.retargeter = retargeter
        self.auto_start_delay_s = float(auto_start_delay_s)
        self.max_consecutive_faults = int(max_consecutive_faults)
        self.log_file = log_file
        self.clock = clock
        self.stats = SessionStats()
        self._tracked_since: float | None = None
        self._consecutive_faults = 0
        self._tick_faulted = False
        self._last_sent: tuple[float, float] | None = None
        self._last_hw_check = 0.0
        self.measured: tuple[float, float] | None = None

    @property
    def anchored(self) -> bool:
        return self.retargeter.state is not HeadState.IDLE

    def _read_measured(self) -> tuple[float, float] | None:
        try:
            self.measured = self.driver.read()
        except HeadBusError as exc:
            self._fault(exc, "read")
            return None
        return self.measured

    def _fault(self, exc: HeadBusError, where: str) -> None:
        """Record a fault. A hardware alert stops at once; others count per tick."""
        if isinstance(exc, HeadAlertError):
            raise exc
        self.stats.faults += 1
        self._tick_faulted = True
        log.warning("Head fault (%s): %s", where, exc)

    def _end_tick(self) -> None:
        if not self._tick_faulted:
            self._consecutive_faults = 0
            return
        self._tick_faulted = False
        self._consecutive_faults += 1
        if self._consecutive_faults > self.max_consecutive_faults:
            raise HeadBusError(
                f"{self._consecutive_faults} consecutive faulty cycles; stopping (head holds)."
            )

    def _maybe_anchor(self, sample: ControllerPairSample, now: float) -> None:
        if not sample.hmd_tracked:
            self._tracked_since = None
            return
        if self._tracked_since is None:
            self._tracked_since = now
            log.info("HMD tracked. Anchoring in %.1f s; keep the head still.",
                     self.auto_start_delay_s)
        if now - self._tracked_since < self.auto_start_delay_s:
            return
        measured = self._read_measured()
        if measured is None:
            return
        if self.retargeter.anchor(np.asarray(sample.device_hmd_pose), measured, now):
            log.info("Head anchored at pan %.1f / tilt %.1f deg.", *measured)

    def _check_hardware(self, now: float) -> None:
        if now - self._last_hw_check < HARDWARE_CHECK_PERIOD_S:
            return
        self._last_hw_check = now
        try:
            errors = self.driver.hardware_errors()
        except HeadBusError as exc:
            self._fault(exc, "hardware status")
            return
        if errors:
            raise HeadBusError(f"head hardware error latched: {errors}")

    def _send(self, step: HeadStep) -> None:
        if step.command_pan_deg is None or step.command_tilt_deg is None:
            return
        command = (step.command_pan_deg, step.command_tilt_deg)
        if self._last_sent is not None and all(
            abs(a - b) < COMMAND_EPSILON_DEG for a, b in zip(command, self._last_sent, strict=True)
        ):
            return
        try:
            self._last_sent = self.driver.command(*command)
            self.stats.commands += 1
        except HeadBusError as exc:
            self._fault(exc, "command")

    def tick(self) -> HeadStep | None:
        now = self.clock()
        self.stats.cycles += 1
        sample = self.tracker.latest()
        self._check_hardware(now)
        if not self.anchored:
            self._maybe_anchor(sample, now)
            self._write_log(now, sample, None)
            self._end_tick()
            return None
        step = self.retargeter.step(
            np.asarray(sample.device_hmd_pose), bool(sample.hmd_tracked), now
        )
        if step.reanchored:
            self.stats.reanchors += 1
            log.info("Head re-anchored at pan %.1f / tilt %.1f deg.",
                     step.command_pan_deg, step.command_tilt_deg)
        if step.state is HeadState.HOLD:
            self.stats.holds += 1
        else:
            self._send(step)
        self._read_measured()
        self._write_log(now, sample, step)
        self._end_tick()
        return step

    def _write_log(
        self, now: float, sample: ControllerPairSample, step: HeadStep | None
    ) -> None:
        if self.log_file is None:
            return
        record = {
            "t_mono_s": round(now, 6),
            "hmd_quat_xyzw": [round(float(v), 6) for v in np.asarray(sample.device_hmd_pose)[3:7]],
            "hmd_tracked": bool(sample.hmd_tracked),
            "state": (step.state.value if step else self.retargeter.state.value),
            "reanchored": bool(step.reanchored) if step else False,
            "rel_yaw_deg": step.rel_yaw_deg if step else None,
            "rel_pitch_deg": step.rel_pitch_deg if step else None,
            "filtered_yaw_deg": step.filtered_yaw_deg if step else None,
            "filtered_pitch_deg": step.filtered_pitch_deg if step else None,
            "cmd_pan_deg": step.command_pan_deg if step else None,
            "cmd_tilt_deg": step.command_tilt_deg if step else None,
            "meas_pan_deg": self.measured[0] if self.measured else None,
            "meas_tilt_deg": self.measured[1] if self.measured else None,
        }
        self.log_file.write(json.dumps(record) + "\n")


def open_log(log_dir: Path, station: str) -> IO[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"head_{station or 'local'}_{stamp}.jsonl"
    log.info("Head log: %s", path)
    return path.open("w", encoding="utf-8", buffering=1)
