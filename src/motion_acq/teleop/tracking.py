"""Tracking-loss debounce and recovery timing shared by real teleop frontends."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from motion_acq.teleop.common import tracking_sample_time_ns


@dataclass(frozen=True)
class TrackingRecoveryConfig:
    lost_debounce_s: float = 0.15
    recover_after_s: float = 2.0
    recover_period_s: float = 5.0


class TrackingRecoveryPolicy:
    """State machine for sustained tracking loss and throttled recovery attempts."""

    def __init__(self, config: TrackingRecoveryConfig | None = None) -> None:
        self.config = config or TrackingRecoveryConfig()
        self.tracking_lost_since: float | None = None
        self.tracking_missing_since: float | None = None
        self.last_recovery_attempt: float | None = None

    @property
    def lost(self) -> bool:
        return self.tracking_lost_since is not None

    def reset(self) -> None:
        self.tracking_missing_since = None
        self.tracking_lost_since = None

    def note_missing(
        self,
        now: float,
        *,
        observed_since: float | None = None,
    ) -> bool:
        """Return True exactly when missing tracking becomes sustained loss."""
        if self.tracking_missing_since is None:
            self.tracking_missing_since = (
                now if observed_since is None else min(now, observed_since)
            )
        if self.tracking_lost_since is not None:
            return False
        if now - self.tracking_missing_since < self.config.lost_debounce_s:
            return False
        self.tracking_lost_since = now
        return True

    def lost_for(self, now: float) -> float:
        if self.tracking_lost_since is None:
            return 0.0
        return now - self.tracking_lost_since

    def should_recover(self, now: float) -> bool:
        if self.tracking_lost_since is None:
            return False
        if self.lost_for(now) < self.config.recover_after_s:
            return False
        if (
            self.last_recovery_attempt is not None
            and now - self.last_recovery_attempt < self.config.recover_period_s
        ):
            return False
        self.last_recovery_attempt = now
        return True


@dataclass(frozen=True)
class TrackingSnapshot:
    """Latest tracking value plus local arrival/freshness timing."""

    sample: Any
    received_at_s: float
    fresh_at_s: float
    source_time_ns: int

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - self.fresh_at_s)


class LatestTrackingSampler:
    """Poll a possibly blocking tracker away from the IK/control loop.

    Only the newest value is retained. Repeated source timestamps do not renew
    freshness, which lets the consumer reject a cached device pose instead of
    applying it late when the SDK or transport resumes.
    """

    def __init__(
        self,
        read: Callable[[], Any],
        *,
        sample_rate_hz: float,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be > 0")
        self._read = read
        self._period_s = 1.0 / float(sample_rate_hz)
        self._latest_lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._latest: TrackingSnapshot | None = None
        self._last_source_time_ns: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-tracking-sampler",
            daemon=True,
        )
        self.last_error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout_s))

    def latest(self) -> TrackingSnapshot | None:
        with self._latest_lock:
            return self._latest

    def try_source_call(self, callback: Callable[[], Any]) -> tuple[bool, Any]:
        """Run tracker recovery only when no sampling call is in flight."""
        if not self._source_lock.acquire(blocking=False):
            return False, None
        try:
            return True, callback()
        finally:
            self._source_lock.release()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                with self._source_lock:
                    sample = self._read()
                arrived = time.monotonic()
                source_time_ns = tracking_sample_time_ns(sample)
                with self._latest_lock:
                    fresh = (
                        self._last_source_time_ns is None
                        or source_time_ns != self._last_source_time_ns
                    )
                    fresh_at = (
                        arrived
                        if fresh or self._latest is None
                        else self._latest.fresh_at_s
                    )
                    self._latest = TrackingSnapshot(
                        sample=sample,
                        received_at_s=arrived,
                        fresh_at_s=fresh_at,
                        source_time_ns=source_time_ns,
                    )
                    self._last_source_time_ns = source_time_ns
                self.last_error = None
            except BaseException as exc:
                self.last_error = exc
                arrived = time.monotonic()
            next_tick += self._period_s
            if next_tick <= arrived:
                next_tick = arrived + self._period_s
            self._stop.wait(max(0.0, next_tick - arrived))
