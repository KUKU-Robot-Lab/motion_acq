"""Shared live-teleop utilities used by sim, real, and recording frontends."""

from __future__ import annotations

import logging
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from typing import Any

import numpy as np

from motion_acq.dataset.raw import pose_to_state_vector
from motion_acq.feetech import zero_gripper_widths
from motion_acq.retargeting.handumi_to_robot import VR_TO_ROBOT
from motion_acq.tracking.transforms import Pose

SIDE_CHOICES = ("left", "right", "both")
# PICO's live tracking stream is 30 Hz. Driving IK faster only retransmits the
# same pose and, because IK limits are expressed per frame, can request joint
# motion faster than a real backend is allowed to stream it.
DEFAULT_TELEOP_FPS = 30
DEFAULT_GRIPPER_SAMPLE_HZ = 200.0


class BestEffortPeriodicWorker:
    """Run disposable peripheral work without blocking the control loop."""

    def __init__(
        self,
        callback: Callable[[], Any],
        *,
        rate_hz: float,
        thread_name: str,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        self.callback = callback
        self.period_s = 1.0 / float(rate_hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        self.error: BaseException | None = None

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def close(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout_s))

    def _run(self) -> None:
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                self.callback()
                now = time.monotonic()
                next_tick += self.period_s
                if next_tick <= now:
                    # Preview work is disposable; never replay missed frames.
                    next_tick = now + self.period_s
                self._stop.wait(max(0.0, next_tick - now))
        except BaseException as exc:
            self.error = exc
            self._stop.set()


class JointMotionDiagnostics:
    """Cheap rolling joint diagnostics shared by live and recorded teleop."""

    def __init__(
        self,
        joint_names: list[str] | tuple[str, ...],
        indices: tuple[int, ...],
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.indices = np.asarray(indices, dtype=np.intp)
        self.reset()

    def reset(self) -> None:
        size = len(self.joint_names)
        self.samples = 0
        self.previous_raw: np.ndarray | None = None
        self.previous_filtered: np.ndarray | None = None
        self.previous_raw_step: np.ndarray | None = None
        self.max_raw_step = np.zeros(size, dtype=np.float32)
        self.max_filtered_step = np.zeros(size, dtype=np.float32)
        self.max_filter_correction = np.zeros(size, dtype=np.float32)
        self.roughness = np.zeros(size, dtype=np.float32)

    def observe(self, raw_q: np.ndarray, filtered_q: np.ndarray) -> None:
        raw = np.asarray(raw_q, dtype=np.float32)
        filtered = np.asarray(filtered_q, dtype=np.float32)
        self.max_filter_correction = np.maximum(
            self.max_filter_correction, np.abs(raw - filtered)
        )
        if self.previous_raw is not None and self.previous_filtered is not None:
            raw_step = raw - self.previous_raw
            filtered_step = filtered - self.previous_filtered
            self.max_raw_step = np.maximum(self.max_raw_step, np.abs(raw_step))
            self.max_filtered_step = np.maximum(
                self.max_filtered_step, np.abs(filtered_step)
            )
            if self.previous_raw_step is not None:
                self.roughness += np.abs(raw_step - self.previous_raw_step)
            self.previous_raw_step = raw_step
        self.previous_raw = raw.copy()
        self.previous_filtered = filtered.copy()
        self.samples += 1

    def summary(self) -> str:
        if self.samples < 2 or self.indices.size == 0:
            return "joint_motion=no fresh samples"
        candidates = self.indices
        roughest = int(candidates[np.argmax(self.roughness[candidates])])
        roughness_per_step = self.roughness[roughest] / max(1, self.samples - 2)
        return (
            f"roughest={self.joint_names[roughest]}, "
            f"IK_roughness={np.rad2deg(roughness_per_step):.2f} deg/step^2, "
            f"IK_step_max={np.rad2deg(self.max_raw_step[roughest]):.2f} deg, "
            "filtered_step_max="
            f"{np.rad2deg(self.max_filtered_step[roughest]):.2f} deg, "
            "filter_correction_max="
            f"{np.rad2deg(self.max_filter_correction[roughest]):.2f} deg"
        )


class AdaptiveJointFilter:
    """Velocity-adaptive low-pass for IK joint targets.

    This is a vectorized One Euro filter. It uses a low cutoff while a joint is
    nearly stationary, then raises that cutoff with measured joint velocity so
    deliberate motion remains responsive. Unlike a deadband, every finite
    target contributes to the output and the command converges exactly when
    the operator stops.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        min_cutoff_hz: float,
        velocity_coefficient: float,
        derivative_cutoff_hz: float,
        filtered_indices: tuple[int, ...] | None = None,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be > 0")
        if min_cutoff_hz <= 0.0:
            raise ValueError("min_cutoff_hz must be > 0")
        if velocity_coefficient < 0.0:
            raise ValueError("velocity_coefficient must be >= 0")
        if derivative_cutoff_hz <= 0.0:
            raise ValueError("derivative_cutoff_hz must be > 0")
        self.nominal_dt_s = 1.0 / float(sample_rate_hz)
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.velocity_coefficient = float(velocity_coefficient)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.filtered_indices = (
            None
            if filtered_indices is None
            else tuple(dict.fromkeys(int(index) for index in filtered_indices))
        )
        if self.filtered_indices and min(self.filtered_indices) < 0:
            raise ValueError("filtered joint indices must be >= 0")
        self._raw_q: np.ndarray | None = None
        self._filtered_q: np.ndarray | None = None
        self._filtered_velocity: np.ndarray | None = None
        self._last_time_s: float | None = None

    def reset(self, q: np.ndarray | None = None) -> None:
        value = None if q is None else np.asarray(q, dtype=np.float32).copy()
        self._raw_q = None if value is None else value.copy()
        self._filtered_q = None if value is None else value.copy()
        self._filtered_velocity = (
            None if value is None else np.zeros_like(value, dtype=np.float32)
        )
        self._last_time_s = None

    def snapshot(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float | None,
    ]:
        """Capture state so a superseded IK solve can be discarded exactly."""
        return (
            None if self._raw_q is None else self._raw_q.copy(),
            None if self._filtered_q is None else self._filtered_q.copy(),
            (
                None
                if self._filtered_velocity is None
                else self._filtered_velocity.copy()
            ),
            self._last_time_s,
        )

    def restore(
        self,
        state: tuple[
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            float | None,
        ],
    ) -> None:
        raw_q, filtered_q, filtered_velocity, last_time_s = state
        self._raw_q = None if raw_q is None else raw_q.copy()
        self._filtered_q = None if filtered_q is None else filtered_q.copy()
        self._filtered_velocity = (
            None if filtered_velocity is None else filtered_velocity.copy()
        )
        self._last_time_s = last_time_s

    def filter(
        self,
        q: np.ndarray,
        now_s: float,
        *,
        exact_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        current = np.asarray(q, dtype=np.float32)
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise ValueError("joint command must be a finite one-dimensional vector")
        if not np.isfinite(now_s):
            raise ValueError("joint command timestamp must be finite")
        if (
            self._raw_q is None
            or self._filtered_q is None
            or self._filtered_velocity is None
            or self._raw_q.shape != current.shape
        ):
            self.reset(current)
            self._last_time_s = float(now_s)
            return current.copy()

        dt_s = self.nominal_dt_s
        if self._last_time_s is not None:
            dt_s = min(max(float(now_s) - self._last_time_s, 1e-6), 0.25)

        raw_velocity = (current - self._raw_q) / dt_s
        derivative_alpha = self._alpha(self.derivative_cutoff_hz, dt_s)
        self._filtered_velocity += derivative_alpha * (
            raw_velocity - self._filtered_velocity
        )
        cutoff_hz = self.min_cutoff_hz + self.velocity_coefficient * np.abs(
            self._filtered_velocity
        )
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt_s)

        if self.filtered_indices is None:
            self._filtered_q += alpha * (current - self._filtered_q)
        else:
            indices = np.asarray(self.filtered_indices, dtype=np.intp)
            if indices.size and int(np.max(indices)) >= current.size:
                raise ValueError("filtered joint index is outside the command vector")
            passthrough = np.ones(current.size, dtype=bool)
            passthrough[indices] = False
            self._filtered_q[indices] += alpha[indices] * (
                current[indices] - self._filtered_q[indices]
            )
            self._filtered_q[passthrough] = current[passthrough]

        if exact_indices:
            exact = np.asarray(exact_indices, dtype=np.intp)
            if int(np.min(exact)) < 0 or int(np.max(exact)) >= current.size:
                raise ValueError("exact joint index is outside the command vector")
            self._filtered_q[exact] = current[exact]
            self._filtered_velocity[exact] = 0.0

        self._raw_q = current.copy()
        self._last_time_s = float(now_s)
        return self._filtered_q.copy()

    @staticmethod
    def _alpha(cutoff_hz: float, dt_s: float) -> float:
        return float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt_s))


class TeleopLoopTimer:
    """Fixed-rate teleop loop timer with real elapsed command dt."""

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        self.interval = 1.0 / float(fps)
        self._last_start: float | None = None

    def tick(self) -> tuple[float, float]:
        now = time.perf_counter()
        if self._last_start is None:
            dt = self.interval
        else:
            dt = max(now - self._last_start, 1e-6)
        self._last_start = now
        return now, min(dt, 2.0 * self.interval)

    def sleep(self, loop_start: float) -> float:
        elapsed = time.perf_counter() - loop_start
        if (delay := self.interval - elapsed) > 0:
            time.sleep(delay)
        return elapsed


class KeyboardSpaceListener:
    """Non-blocking terminal key listener: Space, plus R (reset) and Q (finish).

    The assistant drives recording from the keyboard when the operator's hands
    are busy (gloves, mounted controllers): see teleop_record.
    """

    EPISODE_KEYS = ("r", "q")

    def __init__(self, *, enabled: bool, stop_event: threading.Event | None = None) -> None:
        # One reader owns the terminal: with stop_event set, Esc stops the
        # session here too (a second stdin reader would swallow Space presses).
        self.stop_event = stop_event
        self.enabled = enabled and sys.stdin.isatty()
        self._space = threading.Event()
        self._keys = {key: threading.Event() for key in self.EPISODE_KEYS}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-teleop-space",
            daemon=True,
        )
        self._thread.start()

    def consume_space(self) -> bool:
        if not self._space.is_set():
            return False
        self._space.clear()
        return True

    def press(self, char: str) -> None:
        """Register a key press (used by the reader thread and by tests)."""
        if char == " ":
            self._space.set()
        elif char == "\x1b" and self.stop_event is not None:
            self.stop_event.set()
        elif char.lower() in self._keys:
            self._keys[char.lower()].set()

    def consume_key(self, key: str) -> bool:
        event = self._keys[key]
        if not event.is_set():
            return False
        event.clear()
        return True

    def clear(self) -> None:
        self._space.clear()
        for event in self._keys.values():
            event.clear()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                self.press(sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def enabled_sides(side: str) -> tuple[str, ...]:
    if side == "both":
        return ("left", "right")
    return (side,)


def start_sides(
    anchors: dict[str, dict[str, np.ndarray] | None],
    enabled: tuple[str, ...],
) -> tuple[str, ...]:
    """Return enabled arms that are idle and can be started from Space."""
    return tuple(side for side in enabled if anchors[side] is None)


def tracking_world_map(device: str) -> np.ndarray:
    """Map provider TCP world axes into robot-world axes."""
    return VR_TO_ROBOT if device == "pico" else np.eye(3, dtype=np.float32)


def tracking_heading_from_hmd(device: str) -> bool:
    """True: robot forward = headset heading when an arm starts (Quest, level z-up workspace)."""
    return device == "meta"


def tracking_ready_for_sides(
    source_poses: dict[str, np.ndarray],
    side_tracked: dict[str, bool],
    enabled: tuple[str, ...],
) -> bool:
    """Require a real finite controller pose for every arm being auto-started."""
    return all(
        side_tracked[side]
        and np.isfinite(source_poses[side]).all()
        and float(np.linalg.norm(source_poses[side][:3])) > 1e-6
        for side in enabled
    )


def enabled_tracking_ok(
    side_tracked: dict[str, bool],
    enabled: tuple[str, ...],
) -> bool:
    return all(side_tracked[side] for side in enabled)


def tracking_sample_time_ns(sample: Any) -> int:
    """Stable tracker-frame time for smoothing, preferring device generation time."""
    for value in (
        getattr(sample, "device_time_ns", 0),
        getattr(sample, "aligned_time_ns", 0),
        getattr(sample, "pc_monotonic_ns", 0),
    ):
        if int(value) > 0:
            return int(value)
    return time.monotonic_ns()


def latest_widths(grippers: Any):
    if grippers is None:
        return zero_gripper_widths()
    latest = getattr(grippers, "latest", None)
    if callable(latest):
        sample = latest()
        return (
            zero_gripper_widths()
            if sample is None
            else getattr(sample, "widths", zero_gripper_widths())
        )
    return grippers.read_normalized_widths()


def sample_state(sample, widths=None) -> np.ndarray:
    """16D raw state from a live sample's calibrated TCP poses + gripper widths."""
    left = Pose(sample.left_tcp_pose[:3], sample.left_tcp_pose[3:7])
    right = Pose(sample.right_tcp_pose[:3], sample.right_tcp_pose[3:7])
    left_w = 0.0 if widths is None else widths.left
    right_w = 0.0 if widths is None else widths.right
    return pose_to_state_vector(left, right, left_w, right_w)


def return_home_on_exit(real_env: Any, home_q: np.ndarray, log: logging.Logger) -> None:
    """End pose = start pose: walk the arms home before the motors are disabled.

    Skipped when the arm streamer has failed (following error, CAN loss):
    then nothing can be commanded safely and the arms stop where they are.
    """
    try:
        real_env.check_health()
    except Exception as exc:  # noqa: BLE001 - report and leave the arms alone
        log.error("Arms not returned home (streamer failed: %s).", exc)
        return
    log.info("Returning arms home before exit ...")
    try:
        real_env.move_home(home_q)
        log.info("Arms at home.")
        rest = getattr(real_env, "rest", None)
        if callable(rest):  # stored-path robots go on to rest so the motors can go off safely
            rest(home_q)
            log.info("Arms ready to disable (at rest, or at home for robots without a stored path).")
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - disconnect must still run
        log.error("Arms could not return home / to rest: %s", exc or "interrupted")
