"""In-memory stand-in for the ``openarm_can`` Python module (no CAN, no motors).

Only the calls OpenArmSdkSide makes are implemented. Each motor follows its
MIT position target at a bounded speed, so the real driver code path
(startup read, slow home, streamer, watchdog, following error) runs
unchanged on top of it. With a gravity model per port the motors also sag
like the real PD does: they settle at q* + (tau - G(q)) / kp, so a missing
or wrong gravity feedforward shows up in fake runs (10.04: the real right j7
stopped 0.13 rad short of home without it).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

ARM_DOF = 7
# An unpowered arm reads exactly 0 everywhere; the driver refuses that.
REST_READING = (1e-4, -1e-4, 2e-4, 1e-4, -2e-4, 1e-4, -1e-4)


@dataclass(frozen=True)
class MITParam:
    kp: float
    kd: float
    q: float
    dq: float
    tau: float


class FakeMotor:
    def __init__(self, position: float) -> None:
        self.position = float(position)
        self.target = float(position)
        self.kp = 0.0
        self.tau = 0.0

    def get_position(self) -> float:
        return self.position


class _MotorGroup:
    def __init__(self, motors: list[FakeMotor], owner: FakeOpenArm) -> None:
        self._motors = motors
        self._owner = owner

    def get_motors(self) -> list[FakeMotor]:
        return self._motors

    def mit_control_all(self, params: list[MITParam]) -> None:
        if len(params) != len(self._motors):
            raise ValueError(f"{len(params)} MIT params for {len(self._motors)} motors")
        with self._owner.lock:
            self._owner.commands += 1
            for motor, param in zip(self._motors, params, strict=True):
                motor.target = float(param.q)
                motor.kp = float(param.kp)
                motor.tau = float(param.tau)


class _Gripper:
    def __init__(self, owner: FakeOpenArm) -> None:
        self._owner = owner
        self.position = 0.0

    def set_position(self, value: float) -> None:
        self.position = float(value)
        self._owner.gripper_commands += 1


class FakeOpenArm:
    def __init__(self, port: str, enable_fd: bool, *, start_q: list[float], speed: float,
                 gravity=None) -> None:
        self.port = port
        self.gravity = gravity  # q (7,) -> holding torque (7,), or None
        self.enable_fd = enable_fd
        self.speed = float(speed)
        self.lock = threading.Lock()
        self.enabled = False
        self.commands = 0
        self.gripper_commands = 0
        self.gripper_initialized = False
        self._start_q = list(start_q)
        self._arm: _MotorGroup | None = None
        self._gripper = _Gripper(self)
        self._last = time.monotonic()

    def init_arm_motors(self, motor_types, send_ids, recv_ids) -> None:
        if len(motor_types) != ARM_DOF or len(send_ids) != ARM_DOF or len(recv_ids) != ARM_DOF:
            raise ValueError("OpenArm has 7 arm motors")
        self._arm = _MotorGroup([FakeMotor(q) for q in self._start_q], self)

    def init_gripper_motor(self, *args) -> None:
        self.gripper_initialized = True

    def set_callback_mode_all(self, mode) -> None:
        pass

    def enable_all(self) -> None:
        self.enabled = True

    def disable_all(self) -> None:
        self.enabled = False

    def refresh_all(self) -> None:
        self._advance()

    def recv_all(self, timeout_us: int = 0) -> None:
        self._advance()

    def _advance(self) -> None:
        now = time.monotonic()
        with self.lock:
            dt, self._last = now - self._last, now
            if not self.enabled or self._arm is None:
                return
            step = self.speed * dt
            motors = self._arm.get_motors()
            sag = [0.0] * len(motors)
            if self.gravity is not None:
                load = self.gravity([m.position for m in motors])
                sag = [(m.tau - float(g)) / m.kp if m.kp > 0 else 0.0 for m, g in zip(motors, load, strict=True)]
            for motor, offset in zip(motors, sag, strict=True):
                error = motor.target + offset - motor.position
                motor.position += max(-step, min(step, error))

    def get_arm(self) -> _MotorGroup:
        if self._arm is None:
            raise RuntimeError("init_arm_motors() first")
        return self._arm

    def get_gripper(self) -> _Gripper:
        if not self.gripper_initialized:
            raise RuntimeError("gripper motor was not initialized")
        return self._gripper


@dataclass
class FakeOpenArmSdk:
    """Module-like object: ``sdk.OpenArm(port, fd)``, ``sdk.MITParam``, enums."""

    start_q_by_port: dict[str, list[float]] = field(default_factory=dict)
    speed_rad_s: float = 3.0
    gravity_by_port: dict = field(default_factory=dict)
    arms: dict[str, FakeOpenArm] = field(default_factory=dict)

    MotorType = SimpleNamespace(DM8009="DM8009", DM4340="DM4340", DM4310="DM4310")
    ControlMode = SimpleNamespace(MIT="MIT", POS_FORCE="POS_FORCE")
    CallbackMode = SimpleNamespace(STATE="STATE", PARAM="PARAM")
    MITParam = MITParam

    def OpenArm(self, port: str, enable_fd: bool) -> FakeOpenArm:  # noqa: N802 - SDK name
        arm = FakeOpenArm(
            port, enable_fd,
            # rest, plus the few-millidegree offsets a powered encoder always shows
            start_q=self.start_q_by_port.get(port, list(REST_READING)),
            speed=self.speed_rad_s,
            gravity=self.gravity_by_port.get(port),
        )
        self.arms[port] = arm
        return arm
