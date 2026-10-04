"""OpenArm SDK driver, settings, and streaming for real HandUMI teleop."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import yaml

from motion_acq.real.openarm.gravity import ArmGravity, GravityModelError, load_arm_gravity
from motion_acq.real.openarm.home_path import (
    MAX_PATH_SPEED_RAD_S,
    HomePath,
    HomePathError,
    classify_start,
    load_home_path,
)
from motion_acq.real.streamer import (
    JointStreamer,
    next_periodic_deadline,
    step_toward,
)

log = logging.getLogger(__name__)

SIDES: tuple[str, str] = ("left", "right")
ARM_DOF = 7
SEND_CAN_IDS = tuple(range(0x01, 0x08))
RECV_CAN_IDS = tuple(range(0x11, 0x18))
GRIPPER_SEND_CAN_ID = 0x08
GRIPPER_RECV_CAN_ID = 0x18
DEFAULT_KP = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
DEFAULT_KD = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
JOINT_LIMIT_SNAP_TOLERANCE_RAD = 1e-4
_REPO_ROOT = Path(__file__).resolve().parents[4]
# A failed start retreats along the path only from this close to it.
RETREAT_MAX_OFFSET_RAD = 0.3


@dataclass(frozen=True)
class OpenArmCanSettings:
    """Resolved OpenArm SDK teleop settings from robot defaults + local rig."""

    left_port: str = "can1"
    right_port: str = "can0"
    enable_fd: bool = True
    bitrate: int = 1_000_000
    dbitrate: int = 5_000_000
    # False on stations where another stack owns the CAN links (arm4090 / s2r):
    # validate only, never take the links down to reconfigure them with sudo.
    can_auto_repair: bool = True
    command_rate_hz: float = 100.0
    max_joint_speed_rad_s: float = 1.0
    home_max_joint_speed_rad_s: float = 0.25
    home_timeout_s: float = 30.0
    home_tolerance_rad: float = 0.05
    watchdog_timeout_s: float = 0.15
    following_error_rad: float = 0.35
    # SCHED_FIFO of the streamer thread, as the s2r OpenArm controller_manager (0 = off).
    rt_priority: int = 50
    # False when the J8 gripper motor is absent (e.g. an RH56F1 hand is mounted).
    gripper_enabled: bool = True
    gripper_closed_position_rad: float = 0.0
    gripper_open_position_rad: float = -1.0471975511965976
    left_gripper_closed_position_rad: float | None = None
    left_gripper_open_position_rad: float | None = None
    right_gripper_closed_position_rad: float | None = None
    right_gripper_open_position_rad: float | None = None
    kp: tuple[float, ...] = DEFAULT_KP
    kd: tuple[float, ...] = DEFAULT_KD
    # side -> stored rest<->home path (sim2real); empty: HandUMI slow home
    # (shoulders first) and no rest at the end. See home_path.py.
    home_paths: tuple[tuple[str, str], ...] = ()
    # Gravity torque feedforward in the MIT tau term (sim2real pd model_tau_ff);
    # empty urdf: PD only, as HandUMI. See gravity.py.
    gravity_urdf: str = ""
    gravity_tip_links: tuple[tuple[str, str], ...] = ()
    gravity_scale: tuple[float, ...] = (1.0,) * 7
    gravity_cap_nm: float = 20.0


def load_openarm_settings(
    rig_config: Path,
    robot_real: dict[str, Any] | None = None,
    gripper_calibration_path: Path | None = None,
    robot_name: str = "openarmv1",
) -> OpenArmCanSettings:
    """Combine portable robot defaults with machine-local CAN assignments."""
    robot_real = robot_real or {}
    data: dict[str, Any] = {}
    if rig_config.exists():
        with rig_config.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    # Machine-local CAN/gripper settings live under robots.<robot name> in the
    # station rig (openarmv1 on arm5080, openarm_rh56f1 on arm4090).
    rig_robot = (data.get("robots") or {}).get(robot_name) or {}
    can = rig_robot.get("can") or {}
    rig_gripper = rig_robot.get("gripper") or {}
    control = robot_real.get("control") or {}
    gains = robot_real.get("gains") or {}
    gripper = robot_real.get("gripper") or {}
    gravity = robot_real.get("gravity") or {}
    calibrated: dict[str, Any] = {}
    if gripper_calibration_path is not None and gripper_calibration_path.exists():
        with gripper_calibration_path.open("r", encoding="utf-8") as handle:
            calibrated = yaml.safe_load(handle) or {}

    def calibrated_value(side: str, key: str) -> float | None:
        value = (calibrated.get(side) or {}).get(key)
        return None if value is None else float(value)

    return OpenArmCanSettings(
        left_port=str(can.get("left_port", "can1")),
        right_port=str(can.get("right_port", "can0")),
        enable_fd=bool(can.get("fd", True)),
        bitrate=int(can.get("bitrate", 1_000_000)),
        dbitrate=int(can.get("dbitrate", 5_000_000)),
        can_auto_repair=bool(can.get("auto_repair", True)),
        command_rate_hz=float(control.get("command_rate_hz", 100.0)),
        max_joint_speed_rad_s=float(control.get("max_joint_speed_rad_s", 1.0)),
        home_max_joint_speed_rad_s=float(
            control.get("home_max_joint_speed_rad_s", 0.25)
        ),
        home_timeout_s=float(control.get("home_timeout_s", 30.0)),
        home_tolerance_rad=float(control.get("home_tolerance_rad", 0.05)),
        watchdog_timeout_s=float(control.get("watchdog_timeout_s", 0.15)),
        following_error_rad=float(control.get("following_error_rad", 0.35)),
        rt_priority=int(control.get("rt_priority", 50)),
        gripper_enabled=bool(
            rig_gripper.get("enabled", gripper.get("enabled", True))
        ),
        gripper_closed_position_rad=float(gripper.get("closed_position_rad", 0.0)),
        gripper_open_position_rad=float(
            gripper.get("open_position_rad", -1.0471975511965976)
        ),
        left_gripper_closed_position_rad=calibrated_value(
            "left", "closed_position_rad"
        ),
        left_gripper_open_position_rad=calibrated_value("left", "open_position_rad"),
        right_gripper_closed_position_rad=calibrated_value(
            "right", "closed_position_rad"
        ),
        right_gripper_open_position_rad=calibrated_value(
            "right", "open_position_rad"
        ),
        kp=tuple(float(v) for v in gains.get("kp", DEFAULT_KP)),
        kd=tuple(float(v) for v in gains.get("kd", DEFAULT_KD)),
        home_paths=tuple(
            (str(side), str(_REPO_ROOT / path))
            for side, path in sorted((robot_real.get("home_paths") or {}).items())
        ),
        gravity_urdf=str(_REPO_ROOT / gravity["urdf"]) if gravity.get("urdf") else "",
        gravity_tip_links=tuple(sorted((str(k), str(v)) for k, v in (gravity.get("tip_link") or {}).items())),
        gravity_scale=tuple(float(v) for v in gravity.get("scale", (1.0,) * 7)),
        gravity_cap_nm=float(gravity.get("cap_nm", 20.0)),
    )


def require_openarm_can() -> ModuleType:
    try:
        return importlib.import_module("openarm_can")
    except ImportError as exc:
        raise RuntimeError(
            "OpenArm real support is optional. Install the official C++ library "
            "and run `uv sync --extra openarm`."
        ) from exc


class OpenArmSide(Protocol):
    port: str

    def read_q(self) -> np.ndarray: ...
    def send(self, q: np.ndarray, gripper_opening: float, tau: np.ndarray | None = None) -> None: ...
    def close(self) -> None: ...


class OpenArmSdkSide:
    """One physical arm; all unstable SDK calls are contained here."""

    def __init__(
        self,
        port: str,
        *,
        enable_fd: bool,
        kp: tuple[float, ...],
        kd: tuple[float, ...],
        gripper_closed_position_rad: float = 0.0,
        gripper_open_position_rad: float = -1.0471975511965976,
        gripper_enabled: bool = True,
        sdk: ModuleType | None = None,
    ) -> None:
        self.port = port
        self.sdk = sdk or require_openarm_can()
        self.kp = kp
        self.kd = kd
        self.gripper_closed_position_rad = float(gripper_closed_position_rad)
        self.gripper_open_position_rad = float(gripper_open_position_rad)
        self.gripper_enabled = bool(gripper_enabled)
        motor_types = [
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM8009,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4340,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
            self.sdk.MotorType.DM4310,
        ]
        self.arm = self.sdk.OpenArm(port, enable_fd)
        self.arm.init_arm_motors(motor_types, list(SEND_CAN_IDS), list(RECV_CAN_IDS))
        if self.gripper_enabled:
            self.arm.init_gripper_motor(
                self.sdk.MotorType.DM4310,
                GRIPPER_SEND_CAN_ID,
                GRIPPER_RECV_CAN_ID,
                self.sdk.ControlMode.POS_FORCE,
            )
        self.arm.set_callback_mode_all(self.sdk.CallbackMode.STATE)
        self.arm.enable_all()
        self.arm.recv_all(2_000)

    def read_q(self) -> np.ndarray:
        self.arm.refresh_all()
        time.sleep(0.002)
        self.arm.recv_all(2_000)
        motors = self.arm.get_arm().get_motors()
        values = np.asarray(
            [motor.get_position() for motor in motors], dtype=np.float32
        )
        if values.shape != (ARM_DOF,):
            raise RuntimeError(
                f"OpenArm {self.port} returned {len(values)} joints; expected {ARM_DOF}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"OpenArm {self.port} returned non-finite joint feedback."
            )
        return values

    def read_startup_q(self) -> np.ndarray:
        """Discard cold SDK samples and require a stable measured start pose."""
        samples: list[np.ndarray] = []
        for _ in range(6):
            samples.append(self.read_q())
            time.sleep(0.02)
        recent = np.stack(samples[-3:])
        excursions = np.ptp(recent, axis=0)
        joint = int(np.argmax(excursions))
        if float(excursions[joint]) > 0.1:
            raise RuntimeError(
                f"OpenArm {self.port} startup feedback is unstable at "
                f"joint{joint + 1} ({float(excursions[joint]):.3f} rad span)."
            )
        return np.median(recent, axis=0).astype(np.float32)

    def send(self, q: np.ndarray, gripper_opening: float, tau: np.ndarray | None = None) -> None:
        tau = np.zeros(ARM_DOF) if tau is None else np.asarray(tau, dtype=float)
        params = [
            self.sdk.MITParam(kp, kd, float(target), 0.0, float(t))
            for kp, kd, target, t in zip(self.kp, self.kd, q, tau, strict=True)
        ]
        self.arm.get_arm().mit_control_all(params)
        if self.gripper_enabled:
            opening = float(np.clip(gripper_opening, 0.0, 1.0))
            motor_position = self.gripper_closed_position_rad + opening * (
                self.gripper_open_position_rad - self.gripper_closed_position_rad
            )
            self.arm.get_gripper().set_position(motor_position)
        self.arm.recv_all(500)

    def close(self) -> None:
        self.arm.disable_all()
        self.arm.recv_all(1_000)


SideFactory = Callable[..., OpenArmSide]


def load_gravity_models(settings: OpenArmCanSettings, sides: tuple[str, ...]) -> dict[str, ArmGravity]:
    """Gravity feedforward per side from the robot YAML real.gravity block (empty: off)."""
    if not settings.gravity_urdf:
        return {}
    tips = dict(settings.gravity_tip_links)
    missing = [side for side in sides if side not in tips]
    if missing:
        raise GravityModelError(f"real.gravity.tip_link has no entry for {missing}")
    return {side: load_arm_gravity(Path(settings.gravity_urdf), side, tips[side],
                                   scale=settings.gravity_scale, cap_nm=settings.gravity_cap_nm)
            for side in sides}


class OpenArmJointStreamer(JointStreamer):
    """Velocity-limited latest-target streamer with a stale-command hold."""

    def __init__(
        self,
        arms: dict[str, OpenArmSide],
        settings: OpenArmCanSettings,
        initial_q: dict[str, np.ndarray],
    ) -> None:
        super().__init__(
            command_rate_hz=settings.command_rate_hz,
            thread_name="openarm-sdk-streamer",
            rt_priority=settings.rt_priority,
        )
        self.arms = arms
        self.settings = settings
        self.gravity = load_gravity_models(settings, tuple(arms))
        if self.gravity:
            log.info("OpenArm gravity feedforward on (%s): tau at start %s N m", ", ".join(self.gravity),
                     {side: np.round(model(initial_q[side]), 2).tolist() for side, model in self.gravity.items()})
        self._targets = {side: q.copy() for side, q in initial_q.items()}
        self._commanded = {side: q.copy() for side, q in initial_q.items()}
        self._feedback = {side: q.copy() for side, q in initial_q.items()}
        self._grippers = {side: 0.0 for side in SIDES}
        self._last_target_at = time.monotonic()
        self._max_speed = settings.max_joint_speed_rad_s
        self._waiting_for_targets = False

    def set_max_speed(self, value: float) -> None:
        with self._lock:
            self._max_speed = float(value)

    def set_targets(
        self,
        targets: dict[str, np.ndarray],
        grippers: dict[str, float] | None = None,
    ) -> None:
        self.raise_if_failed()
        with self._lock:
            for side, target in targets.items():
                q = np.asarray(target, dtype=np.float32)
                if q.shape != (ARM_DOF,) or not np.all(np.isfinite(q)):
                    raise ValueError(
                        f"Invalid OpenArm target for {side}: shape={q.shape}"
                    )
                self._targets[side] = q.copy()
            if grippers:
                self._grippers.update(
                    {
                        side: float(np.clip(value, 0.0, 1.0))
                        for side, value in grippers.items()
                    }
                )
            self._last_target_at = time.monotonic()

    def hold(self) -> dict[str, np.ndarray]:
        self.raise_if_failed()
        with self._lock:
            held = {side: q.copy() for side, q in self._feedback.items()}
            self._targets = {side: q.copy() for side, q in held.items()}
            self._commanded = {side: q.copy() for side, q in held.items()}
            self._last_target_at = time.monotonic()
        return held

    def feedback(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: q.copy() for side, q in self._feedback.items()}

    def wait_until_targets(self, *, timeout_s: float, tolerance_rad: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            expected = {side: self._targets[side].copy() for side in self.arms}
            self._waiting_for_targets = True
        try:
            while True:
                self.raise_if_failed()
                with self._lock:
                    joint_errors = {
                        side: np.abs(self._feedback[side] - expected[side])
                        for side in self.arms
                    }
                    worst_side = max(
                        joint_errors,
                        key=lambda side: float(np.max(joint_errors[side])),
                        default=None,
                    )
                    worst_joint = (
                        int(np.argmax(joint_errors[worst_side]))
                        if worst_side is not None
                        else 0
                    )
                    max_error = (
                        float(joint_errors[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                    measured = (
                        float(self._feedback[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                    target = (
                        float(expected[worst_side][worst_joint])
                        if worst_side is not None
                        else 0.0
                    )
                if max_error <= tolerance_rad:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"OpenArm home timeout: {worst_side} joint{worst_joint + 1} "
                        f"error={max_error:.3f} rad, measured={measured:.3f} rad, "
                        f"target={target:.3f} rad."
                    )
                time.sleep(0.05)
        finally:
            with self._lock:
                self._waiting_for_targets = False
                self._last_target_at = time.monotonic()

    def _run(self) -> None:
        period = 1.0 / self.command_rate_hz
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    if (
                        not self._waiting_for_targets
                        and now - self._last_target_at
                        > self.settings.watchdog_timeout_s
                    ):
                        self._targets = {
                            side: q.copy() for side, q in self._commanded.items()
                        }
                    max_step = self._max_speed * period
                    commands = {
                        side: step_toward(
                            self._commanded[side], self._targets[side], max_step
                        ).astype(np.float32)
                        for side in self.arms
                    }
                    grippers = self._grippers.copy()
                    self._commanded = {side: q.copy() for side, q in commands.items()}

                feedback: dict[str, np.ndarray] = {}
                with self._lock:
                    last_feedback = {side: q.copy() for side, q in self._feedback.items()}
                for side, arm in self.arms.items():
                    model = self.gravity.get(side)
                    tau = None if model is None else model(last_feedback[side])
                    arm.send(commands[side], grippers[side], tau)
                    feedback[side] = arm.read_q()
                    joint_errors = np.abs(feedback[side] - commands[side])
                    joint = int(np.argmax(joint_errors))
                    error = float(joint_errors[joint])
                    if error > self.settings.following_error_rad:
                        raise RuntimeError(
                            f"OpenArm {side} joint{joint + 1} following error "
                            f"{error:.3f} rad exceeds "
                            f"{self.settings.following_error_rad:.3f} rad."
                        )
                with self._lock:
                    self._feedback = feedback
                now = time.monotonic()
                next_tick = next_periodic_deadline(next_tick, period, now)
                if (remaining := next_tick - now) > 0:
                    self._stop.wait(remaining)
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            log.error("OpenArm command streamer failed: %s", exc)


class OpenArmCanEnvironment:
    """Bimanual OpenArm backend implementing the generic teleop contract."""

    def __init__(
        self,
        settings: OpenArmCanSettings,
        *,
        side_factory: SideFactory = OpenArmSdkSide,
        active_sides: tuple[str, ...] = SIDES,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not active_sides or any(side not in SIDES for side in active_sides):
            raise ValueError(f"Invalid OpenArm active sides: {active_sides}")
        self.settings = settings
        self.side_factory = side_factory
        self.active_sides = tuple(dict.fromkeys(active_sides))
        self.joint_limits = joint_limits or {}
        self.arms: dict[str, OpenArmSide] = {}
        self.streamer: OpenArmJointStreamer | None = None
        self._path_progress: dict[str, tuple[np.ndarray, int]] = {}
        self._last_limit_warning_at = {side: 0.0 for side in SIDES}

    def connect(self) -> None:
        if self.arms:
            return
        ports = {"left": self.settings.left_port, "right": self.settings.right_port}
        for side in self.active_sides:
            port = ports[side]
            closed = getattr(self.settings, f"{side}_gripper_closed_position_rad")
            open_ = getattr(self.settings, f"{side}_gripper_open_position_rad")
            log.info("Connecting OpenArm %s on %s.", side, port)
            self.arms[side] = self.side_factory(
                port,
                enable_fd=self.settings.enable_fd,
                kp=self.settings.kp,
                kd=self.settings.kd,
                gripper_enabled=self.settings.gripper_enabled,
                gripper_closed_position_rad=(
                    self.settings.gripper_closed_position_rad
                    if closed is None
                    else closed
                ),
                gripper_open_position_rad=(
                    self.settings.gripper_open_position_rad
                    if open_ is None
                    else open_
                ),
            )

    def prepare(self, *, repair: bool = True) -> None:
        from motion_acq.real.can_setup import ensure_can_fd_interfaces_ready

        ensure_can_fd_interfaces_ready(
            [
                self.settings.left_port if side == "left" else self.settings.right_port
                for side in self.active_sides
            ],
            bitrate=self.settings.bitrate,
            dbitrate=self.settings.dbitrate,
            repair=repair and self.settings.can_auto_repair,
        )

    def _split_side_q(
        self, q: np.ndarray, joint_names: list[str], side: str
    ) -> np.ndarray:
        names = [f"openarm_{side}_joint{i}" for i in range(1, ARM_DOF + 1)]
        values = np.asarray(
            [q[joint_names.index(name)] for name in names], dtype=np.float32
        )
        for index, (name, value) in enumerate(zip(names, values, strict=True)):
            limits = self.joint_limits.get(name)
            if limits is None:
                continue
            lower, upper = limits
            scalar = float(value)
            if (
                scalar < lower - JOINT_LIMIT_SNAP_TOLERANCE_RAD
                or scalar > upper + JOINT_LIMIT_SNAP_TOLERANCE_RAD
            ):
                raise ValueError(
                    f"OpenArm target {name}={scalar:.8f} is outside "
                    f"URDF limits [{lower:.8f}, {upper:.8f}]."
                )
            values[index] = np.clip(scalar, lower, upper)
        return values

    def _split_q(self, q: np.ndarray, joint_names: list[str]) -> dict[str, np.ndarray]:
        return {
            side: self._split_side_q(q, joint_names, side)
            for side in self.active_sides
        }

    @staticmethod
    def _merge_q(
        arm_q: dict[str, np.ndarray], base_q: np.ndarray, joint_names: list[str]
    ) -> np.ndarray:
        q = np.asarray(base_q, dtype=np.float32).copy()
        for side, values in arm_q.items():
            for i, value in enumerate(values, start=1):
                q[joint_names.index(f"openarm_{side}_joint{i}")] = value
        return q

    def home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if not self.arms:
            raise RuntimeError("connect() before home()")
        initial = {
            side: getattr(arm, "read_startup_q", arm.read_q)()
            for side, arm in self.arms.items()
        }
        for side, measured in initial.items():
            log.info(
                "OpenArm %s measured startup joints (deg): %s",
                side,
                np.round(np.rad2deg(measured), 1).tolist(),
            )
        paths = self._home_paths()
        modes: dict[str, str] = {}
        if paths:
            targets = self._split_q(q, joint_names)
            for side in self.active_sides:
                modes[side] = classify_start(
                    initial[side], paths[side], targets[side], self.settings.home_tolerance_rad
                )
            log.info("OpenArm start: %s (stored sim2real path from rest; the RH56F1 must be closed)",
                     ", ".join(f"{side} at {mode}" for side, mode in modes.items()))
        self.streamer = OpenArmJointStreamer(self.arms, self.settings, initial)
        self.streamer.start()
        from_rest = {side: paths[side].q for side, mode in modes.items() if mode == "rest"}
        self._path_progress = {}
        try:
            if from_rest:
                self._play_paths(from_rest, next(iter(paths.values())).dt, "rest -> home")
            self.move_home(q, joint_names)
        except Exception as exc:
            if paths:
                self._retreat_to_rest(paths, exc)
            raise

    def _retreat_to_rest(self, paths: dict[str, HomePath], cause: BaseException) -> None:
        """A start that failed after leaving rest goes back along the path it came.

        Disabling the motors anywhere but rest drops the arm (10.04 right arm,
        home timeout). Only possible while the streamer is healthy; otherwise
        the arm stays where it is until the motors are switched off.
        """
        assert self.streamer is not None
        try:
            self.streamer.raise_if_failed()
        except RuntimeError as exc:
            log.error("OpenArm start failed (%s) and the streamer is down (%s): the arm cannot be "
                      "brought back to rest and will drop when the motors go off. Support it.", cause, exc)
            return
        back = {}
        feedback = self.streamer.feedback()
        for side, path in paths.items():
            sent, k = self._path_progress.get(side, (path.q, len(path.q) - 1))
            route = sent[: k + 1][::-1]
            off = float(np.abs(feedback[side] - route[0]).max())
            if off > RETREAT_MAX_OFFSET_RAD:
                log.error("OpenArm %s is %.2f rad off the stored path; not retreating, holding in place. "
                          "Support the arm before the motors go off.", side, off)
                self.streamer.hold()
                return
            back[side] = route
        log.error("OpenArm start failed (%s); returning to rest along the stored path.", cause)
        try:
            self._play_paths(back, next(iter(paths.values())).dt, "retreat -> rest")
            log.info("OpenArm back at rest after the failed start.")
        except Exception as exc:  # noqa: BLE001 - report both, the caller re-raises the cause
            log.error("OpenArm retreat to rest failed too (%s); support the arm.", exc)

    def _home_paths(self) -> dict[str, HomePath]:
        configured = dict(self.settings.home_paths)
        if not configured:
            return {}
        missing = [side for side in self.active_sides if side not in configured]
        if missing:
            raise HomePathError(f"no stored home path for the {missing} arm(s)")
        return {side: load_home_path(Path(configured[side]), side) for side in self.active_sides}

    def _play_paths(self, paths: dict[str, np.ndarray], dt: float, label: str) -> None:
        """Stream stored joint paths (one per side, same step) and wait for the last point."""
        if self.streamer is None:
            raise RuntimeError("home() before _play_paths()")
        steps = max(len(path) for path in paths.values())
        log.info("OpenArm %s: following the stored path (%s), %.1f s.", ", ".join(paths), label, (steps - 1) * dt)
        # The path already respects 0.3 rad/s; the streamer limit only must not lag behind it.
        self.streamer.set_max_speed(max(self.settings.home_max_joint_speed_rad_s, 1.5 * MAX_PATH_SPEED_RAD_S))
        try:
            start = time.monotonic()
            for k in range(steps):
                self.streamer.set_targets(
                    {side: path[min(k, len(path) - 1)].astype(np.float32) for side, path in paths.items()},
                    {side: 0.0 for side in paths},
                )
                self._path_progress = {side: (path, min(k, len(path) - 1)) for side, path in paths.items()}
                delay = start + (k + 1) * dt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_rad=self.settings.home_tolerance_rad,
            )
        finally:
            self.streamer.set_max_speed(self.settings.max_joint_speed_rad_s)

    def rest(self, q: np.ndarray, joint_names: list[str]) -> None:
        """End of a session: from home, back along the stored path to rest.

        No-op without stored paths (HandUMI robots end at home). Call after
        move_home(); refuses if an arm is not at home.
        """
        paths = self._home_paths()
        if not paths:
            return
        if self.streamer is None:
            raise RuntimeError("home() before rest()")
        feedback = self.streamer.feedback()
        targets = self._split_q(q, joint_names)
        for side in self.active_sides:
            off = float(np.abs(feedback[side] - targets[side]).max())
            if off > self.settings.home_tolerance_rad:
                raise HomePathError(f"OpenArm {side} is {off:.3f} rad from home; not starting the path to rest")
        self._play_paths({side: path.q[::-1] for side, path in paths.items()},
                         next(iter(paths.values())).dt, "home -> rest")

    def move_home(self, q: np.ndarray, joint_names: list[str]) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before move_home()")
        targets = self._split_q(q, joint_names)
        for side, target in targets.items():
            log.info(
                "Moving OpenArm %s slowly to home (deg): %s at %.1f deg/s max.",
                side,
                np.round(np.rad2deg(target), 1).tolist(),
                float(np.rad2deg(self.settings.home_max_joint_speed_rad_s)),
            )
        self.streamer.set_max_speed(self.settings.home_max_joint_speed_rad_s)
        try:
            if any(np.any(np.abs(target[:3]) > 1e-4) for target in targets.values()):
                feedback = self.streamer.feedback()
                clearance_targets = {
                    side: np.concatenate((target[:3], feedback[side][3:])).astype(
                        np.float32
                    )
                    for side, target in targets.items()
                }
                log.info(
                    "Opening OpenArm shoulders first to clear the center structure."
                )
                self.streamer.set_targets(
                    clearance_targets,
                    {side: 0.0 for side in self.active_sides},
                )
                self.streamer.wait_until_targets(
                    timeout_s=self.settings.home_timeout_s,
                    tolerance_rad=self.settings.home_tolerance_rad,
                )
            self.streamer.set_targets(
                targets,
                {side: 0.0 for side in self.active_sides},
            )
            self.streamer.wait_until_targets(
                timeout_s=self.settings.home_timeout_s,
                tolerance_rad=self.settings.home_tolerance_rad,
            )
        finally:
            self.streamer.set_max_speed(self.settings.max_joint_speed_rad_s)

    def command(
        self,
        q: np.ndarray,
        joint_names: list[str],
        gripper_openings: dict[str, float],
    ) -> None:
        if self.streamer is None:
            raise RuntimeError("home() before command()")
        targets: dict[str, np.ndarray] = {}
        for side in self.active_sides:
            try:
                targets[side] = self._split_side_q(q, joint_names, side)
            except ValueError as exc:
                now = time.monotonic()
                if now - self._last_limit_warning_at[side] >= 1.0:
                    log.warning(
                        "%s Holding the previous safe %s-arm target.", exc, side
                    )
                    self._last_limit_warning_at[side] = now
        self.streamer.set_targets(targets, gripper_openings)

    def hold(self, base_q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        if self.streamer is None:
            raise RuntimeError("home() before hold()")
        return self._merge_q(self.streamer.hold(), base_q, joint_names)

    def check_health(self) -> None:
        if self.streamer is not None:
            self.streamer.raise_if_failed()

    def close(self) -> None:
        error: BaseException | None = None
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except BaseException as exc:
                error = exc
        for side, arm in list(self.arms.items()):
            try:
                arm.close()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                log.warning("Failed to disable OpenArm %s: %s", side, exc)
        self.arms.clear()
        self.streamer = None
        if error is not None:
            raise error


__all__ = [
    "ARM_DOF",
    "DEFAULT_KD",
    "DEFAULT_KP",
    "GRIPPER_RECV_CAN_ID",
    "GRIPPER_SEND_CAN_ID",
    "JOINT_LIMIT_SNAP_TOLERANCE_RAD",
    "OpenArmCanEnvironment",
    "OpenArmCanSettings",
    "OpenArmJointStreamer",
    "OpenArmSdkSide",
    "OpenArmSide",
    "RECV_CAN_IDS",
    "SEND_CAN_IDS",
    "SIDES",
    "SideFactory",
    "load_openarm_settings",
    "require_openarm_can",
]
