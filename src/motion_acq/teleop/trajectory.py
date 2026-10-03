"""Interpolated, time-driven joint command playback for live teleoperation."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from motion_acq.real.streamer import next_periodic_deadline

DEFAULT_GRIPPER_MAX_RATE_PER_S = 10.0


@dataclass(frozen=True)
class JointCommand:
    """One timestamped IK result in the PC monotonic clock domain."""

    time_s: float
    q: np.ndarray
    openings: dict[str, float]


@dataclass(frozen=True)
class CommandPlayerStats:
    """Observable timing of the fixed-rate output thread."""

    elapsed_s: float
    published_commands: int
    effective_rate_hz: float
    missed_deadlines: int
    max_lateness_ms: float
    max_write_ms: float
    latest_target_age_ms: float
    interpolated_commands: int
    extrapolated_commands: int
    held_commands: int


class DelayedJointCommandBuffer:
    """Interpolate timestamped IK results at ``now - delay``.

    The buffer retains the sample immediately before the playback cursor and
    all newer samples.  Joint positions and normalized gripper openings are
    linearly interpolated. If tracking does not provide the right-hand
    bracket in time, playback uses an optional bounded velocity bridge and
    then safely holds the newest predicted command.
    """

    def __init__(
        self,
        delay_s: float,
        *,
        max_extrapolation_s: float = 0.0,
        max_commands: int = 128,
    ) -> None:
        if delay_s < 0.0:
            raise ValueError("delay_s must be >= 0")
        if max_extrapolation_s < 0.0:
            raise ValueError("max_extrapolation_s must be >= 0")
        if max_commands < 2:
            raise ValueError("max_commands must be >= 2")
        self.delay_s = float(delay_s)
        self.max_extrapolation_s = float(max_extrapolation_s)
        self._commands: deque[JointCommand] = deque(maxlen=max_commands)
        self._lock = threading.Lock()

    def reset(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        command = self._command(q, openings, time_s)
        with self._lock:
            self._commands.clear()
            self._commands.append(command)

    def push(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        command = self._command(q, openings, time_s)
        with self._lock:
            if self._commands and command.time_s <= self._commands[-1].time_s:
                if command.time_s == self._commands[-1].time_s:
                    self._commands[-1] = command
                    return
                raise ValueError("joint command timestamps must be monotonic")
            self._commands.append(command)

    def sample(self, now_s: float) -> tuple[np.ndarray, dict[str, float]] | None:
        result = self.sample_with_status(now_s)
        if result is None:
            return None
        q, openings, _ = result
        return q, openings

    def sample_with_status(
        self, now_s: float
    ) -> tuple[np.ndarray, dict[str, float], str] | None:
        """Sample a command and report interpolation, extrapolation, or hold."""
        playback_s = float(now_s) - self.delay_s
        with self._lock:
            if not self._commands:
                return None
            while len(self._commands) >= 3 and self._commands[1].time_s <= playback_s:
                self._commands.popleft()
            first = self._commands[0]
            if playback_s <= first.time_s or len(self._commands) == 1:
                return first.q.copy(), first.openings.copy(), "hold"
            second = self._commands[1]
            if playback_s >= second.time_s:
                # Bridge a short producer underflow at the measured IK
                # velocity.  The horizon is deliberately bounded: it removes
                # the 30 Hz staircase without letting stale tracking run away.
                source_dt = second.time_s - first.time_s
                underflow_s = playback_s - second.time_s
                horizon_s = min(underflow_s, self.max_extrapolation_s)
                if horizon_s <= 0.0:
                    return second.q.copy(), second.openings.copy(), "hold"
                q = second.q + (horizon_s / source_dt) * (second.q - first.q)
                status = (
                    "extrapolate"
                    if underflow_s <= self.max_extrapolation_s
                    else "hold"
                )
                return q.astype(np.float32), second.openings.copy(), status
            fraction = (playback_s - first.time_s) / (second.time_s - first.time_s)
            q = first.q + fraction * (second.q - first.q)
            sides = first.openings.keys() | second.openings.keys()
            openings: dict[str, float] = {}
            for side in sides:
                first_value = first.openings.get(side)
                second_value = second.openings.get(side)
                if first_value is None:
                    first_value = second_value
                if second_value is None:
                    second_value = first_value
                assert first_value is not None and second_value is not None
                openings[side] = first_value + fraction * (second_value - first_value)
            return q.astype(np.float32), openings, "interpolate"

    @staticmethod
    def _command(
        q: np.ndarray,
        openings: Mapping[str, float],
        time_s: float,
    ) -> JointCommand:
        q_value = np.asarray(q, dtype=np.float32).copy()
        if not np.all(np.isfinite(q_value)):
            raise ValueError("joint command contains non-finite values")
        if not np.isfinite(time_s):
            raise ValueError("joint command timestamp must be finite")
        return JointCommand(
            time_s=float(time_s),
            q=q_value,
            openings={side: float(value) for side, value in openings.items()},
        )


class DelayedJointCommandPlayer:
    """Read a delayed command buffer and publish it at a fixed rate.

    EMA is applied only to arm joints. Gripper openings already arrive as
    calibrated operator commands and must not acquire additional response lag.
    """

    def __init__(
        self,
        write: Callable[[np.ndarray, dict[str, float]], None],
        *,
        command_rate_hz: float,
        delay_s: float,
        max_extrapolation_s: float = 0.0,
        ema_time_constant_s: float = 0.0,
        gripper_max_rate_per_s: float = 0.0,
    ) -> None:
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        if ema_time_constant_s < 0.0:
            raise ValueError("ema_time_constant_s must be >= 0")
        if gripper_max_rate_per_s < 0.0:
            raise ValueError("gripper_max_rate_per_s must be >= 0")
        self.command_rate_hz = float(command_rate_hz)
        self.ema_time_constant_s = float(ema_time_constant_s)
        self.gripper_max_rate_per_s = float(gripper_max_rate_per_s)
        self.buffer = DelayedJointCommandBuffer(
            delay_s,
            max_extrapolation_s=max_extrapolation_s,
        )
        self._write = write
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._latest_lock = threading.Lock()
        self._latest: tuple[np.ndarray, dict[str, float]] | None = None
        self._target_openings_lock = threading.Lock()
        self._target_openings: dict[str, float] = {}
        self._filtered_q: np.ndarray | None = None
        self._filtered_openings: dict[str, float] = {}
        self._joint_rate_limits_lock = threading.Lock()
        self._joint_rate_limits: dict[int, float] = {}
        self._stats_lock = threading.Lock()
        self._started_at_s = 0.0
        self._last_target_at_s = 0.0
        self._published_commands = 0
        self._missed_deadlines = 0
        self._max_lateness_s = 0.0
        self._max_write_s = 0.0
        self._interpolated_commands = 0
        self._extrapolated_commands = 0
        self._held_commands = 0

    def start(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("joint command player is already running")
        started_at = time.perf_counter()
        self.buffer.reset(
            q,
            openings,
            time_s=started_at if time_s is None else time_s,
        )
        with self._stats_lock:
            self._started_at_s = started_at
            self._last_target_at_s = started_at
            self._published_commands = 0
            self._missed_deadlines = 0
            self._max_lateness_s = 0.0
            self._max_write_s = 0.0
            self._interpolated_commands = 0
            self._extrapolated_commands = 0
            self._held_commands = 0
        with self._target_openings_lock:
            self._target_openings = {
                side: float(value) for side, value in openings.items()
            }
        self._stop.clear()
        self._error = None
        with self._latest_lock:
            self._latest = None
        self._filtered_q = None
        self._filtered_openings = {}
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-delayed-joint-player",
            daemon=True,
        )
        self._thread.start()

    def update_openings(self, openings: Mapping[str, float]) -> None:
        """Update grippers without fabricating a duplicate arm waypoint."""
        self.raise_if_failed()
        with self._target_openings_lock:
            self._target_openings = {
                side: float(value) for side, value in openings.items()
            }

    def push(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        self.raise_if_failed()
        self.buffer.push(q, openings, time_s=time_s)
        with self._stats_lock:
            self._last_target_at_s = time.perf_counter()
        with self._target_openings_lock:
            self._target_openings = {
                side: float(value) for side, value in openings.items()
            }

    def limit_joint_rates(
        self,
        indices: tuple[int, ...],
        max_rate_rad_s: float,
    ) -> None:
        """Rate-limit selected joints until their limit is explicitly cleared."""
        if max_rate_rad_s <= 0.0:
            raise ValueError("max_rate_rad_s must be > 0")
        with self._joint_rate_limits_lock:
            for index in indices:
                if index < 0:
                    raise ValueError("joint indices must be >= 0")
                self._joint_rate_limits[int(index)] = float(max_rate_rad_s)

    def clear_joint_rate_limits(self, indices: tuple[int, ...]) -> None:
        with self._joint_rate_limits_lock:
            for index in indices:
                self._joint_rate_limits.pop(int(index), None)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("delayed joint command player failed") from self._error

    def latest(self) -> tuple[np.ndarray, dict[str, float]] | None:
        """Return the last command successfully handed to the output callback."""
        with self._latest_lock:
            if self._latest is None:
                return None
            q, openings = self._latest
            return q.copy(), openings.copy()

    def stats(self) -> CommandPlayerStats:
        now_s = time.perf_counter()
        with self._stats_lock:
            elapsed_s = max(0.0, now_s - self._started_at_s)
            published = self._published_commands
            return CommandPlayerStats(
                elapsed_s=elapsed_s,
                published_commands=published,
                effective_rate_hz=(published / elapsed_s if elapsed_s > 0.0 else 0.0),
                missed_deadlines=self._missed_deadlines,
                max_lateness_ms=self._max_lateness_s * 1000.0,
                max_write_ms=self._max_write_s * 1000.0,
                latest_target_age_ms=max(0.0, now_s - self._last_target_at_s) * 1000.0,
                interpolated_commands=self._interpolated_commands,
                extrapolated_commands=self._extrapolated_commands,
                held_commands=self._held_commands,
            )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        period_s = 1.0 / self.command_rate_hz
        alpha = (
            1.0
            if self.ema_time_constant_s == 0.0
            else float(1.0 - np.exp(-period_s / self.ema_time_constant_s))
        )
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                # Sample in actual time.  Sampling an old scheduled deadline
                # after a process stall would briefly replay stale motion.
                tick_s = time.perf_counter()
                lateness_s = max(0.0, tick_s - next_tick)
                sampled = self.buffer.sample_with_status(tick_s)
                write_s = 0.0
                playback_status = "hold"
                if sampled is not None:
                    q, _delayed_openings, playback_status = sampled
                    with self._target_openings_lock:
                        immediate_openings = self._target_openings.copy()
                    filtered = self._smooth(
                        q,
                        immediate_openings,
                        alpha=alpha,
                        gripper_max_step=self.gripper_max_rate_per_s * period_s,
                        period_s=period_s,
                    )
                    write_start_s = time.perf_counter()
                    self._write(*filtered)
                    write_s = time.perf_counter() - write_start_s
                    with self._latest_lock:
                        self._latest = (filtered[0].copy(), filtered[1].copy())
                now_s = time.perf_counter()
                regular_deadline = next_tick + period_s
                missed = (
                    int((now_s - regular_deadline) // period_s) + 1
                    if regular_deadline <= now_s
                    else 0
                )
                with self._stats_lock:
                    if sampled is not None:
                        self._published_commands += 1
                        self._max_write_s = max(self._max_write_s, write_s)
                        if playback_status == "interpolate":
                            self._interpolated_commands += 1
                        elif playback_status == "extrapolate":
                            self._extrapolated_commands += 1
                        else:
                            self._held_commands += 1
                    self._missed_deadlines += missed
                    self._max_lateness_s = max(self._max_lateness_s, lateness_s)
                next_tick = next_periodic_deadline(next_tick, period_s, now_s)
                remaining_s = next_tick - now_s
                if remaining_s > 0.0:
                    self._stop.wait(remaining_s)
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def _smooth(
        self,
        q: np.ndarray,
        openings: dict[str, float],
        *,
        alpha: float,
        gripper_max_step: float = 0.0,
        period_s: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        previous_q = None if self._filtered_q is None else self._filtered_q.copy()
        if previous_q is None or alpha >= 1.0:
            target_q = q.copy()
        else:
            target_q = previous_q + alpha * (q - previous_q)
        if previous_q is not None:
            with self._joint_rate_limits_lock:
                joint_rate_limits = self._joint_rate_limits.copy()
            for index, max_rate_rad_s in joint_rate_limits.items():
                if index >= target_q.size:
                    raise ValueError(
                        "rate-limited joint index is outside command vector"
                    )
                max_step = max_rate_rad_s * period_s
                target_q[index] = previous_q[index] + np.clip(
                    target_q[index] - previous_q[index], -max_step, max_step
                )
        self._filtered_q = target_q
        if not self._filtered_openings or gripper_max_step <= 0.0:
            self._filtered_openings = openings.copy()
        else:
            for side, target in openings.items():
                previous = self._filtered_openings.get(side, target)
                delta = float(
                    np.clip(
                        target - previous,
                        -gripper_max_step,
                        gripper_max_step,
                    )
                )
                self._filtered_openings[side] = previous + delta
        return self._filtered_q.copy(), self._filtered_openings.copy()


class TeleopCommandStream:
    """Own the common 30 Hz IK -> interpolated 100 Hz output lifecycle."""

    def __init__(
        self,
        write: Callable[[np.ndarray, dict[str, float]], None],
        *,
        command_rate_hz: float,
        delay_s: float,
        ema_time_constant_s: float,
        max_extrapolation_s: float = 0.0,
    ) -> None:
        self.player = DelayedJointCommandPlayer(
            write,
            command_rate_hz=command_rate_hz,
            delay_s=delay_s,
            max_extrapolation_s=max_extrapolation_s,
            ema_time_constant_s=ema_time_constant_s,
            gripper_max_rate_per_s=DEFAULT_GRIPPER_MAX_RATE_PER_S,
        )

    def submit(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
        active: bool,
        new_epoch: bool = False,
    ) -> None:
        """Submit one IK result, resetting interpolation at a fresh anchor."""
        if new_epoch:
            self.player.stop()
            self.player.start(q, openings, time_s=time_s)
        elif active:
            if self.player.running:
                self.player.push(q, openings, time_s=time_s)
            else:
                self.player.start(q, openings, time_s=time_s)

    def stop(self) -> None:
        self.player.stop()

    def limit_joint_rates(
        self,
        indices: tuple[int, ...],
        max_rate_rad_s: float,
    ) -> None:
        self.player.limit_joint_rates(indices, max_rate_rad_s)

    def clear_joint_rate_limits(self, indices: tuple[int, ...]) -> None:
        self.player.clear_joint_rate_limits(indices)

    def update_openings(self, openings: Mapping[str, float]) -> None:
        if self.player.running:
            self.player.update_openings(openings)

    def latest(self) -> tuple[np.ndarray, dict[str, float]] | None:
        return self.player.latest()

    def stats(self) -> CommandPlayerStats:
        return self.player.stats()

    @property
    def running(self) -> bool:
        return self.player.running


__all__ = [
    "DEFAULT_GRIPPER_MAX_RATE_PER_S",
    "CommandPlayerStats",
    "DelayedJointCommandBuffer",
    "DelayedJointCommandPlayer",
    "JointCommand",
    "TeleopCommandStream",
]
