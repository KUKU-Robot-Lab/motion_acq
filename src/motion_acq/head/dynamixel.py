"""Dynamixel pan/tilt head driver (XC330, Protocol 2.0) with a fake bus.

Register addresses, the tick convention and the setup order follow the
verified sim2real head tools (scripts/head_home.py, head_position_hold_node.py):

    torque off -> operating mode -> position I gain -> profile -> goal = present -> torque on

Writing the operating mode resets the gains to that mode's defaults, so the I
gain must come after it (sim2real 2026-09-01: tilt sagged +1.56 deg when the
order was reversed). Seeding goal = present before torque on keeps the head
still at start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_POSITION_I_GAIN = 82
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_OFF = 0
TORQUE_ON = 1
TICK_MAX = 4095
HARDWARE_ERROR_BITS = {
    0x01: "input_voltage",
    0x04: "overheating",
    0x08: "motor_encoder",
    0x10: "electrical_shock",
    0x20: "overload",
}


class HeadBusError(RuntimeError):
    """A Dynamixel packet exchange failed or the device reported an error."""


def deg_to_tick(deg: float) -> int:
    """sim2real convention: signed degrees -> position tick."""
    return round((float(deg) + 180.0) / 360.0 * TICK_MAX)


def tick_to_deg(tick: int) -> float:
    return int(tick) / TICK_MAX * 360.0 - 180.0


def decode_hardware_errors(status: int) -> tuple[str, ...]:
    names = [name for bit, name in HARDWARE_ERROR_BITS.items() if status & bit]
    unknown = status & ~sum(HARDWARE_ERROR_BITS)
    if unknown:
        names.append(f"unknown_0x{unknown:02x}")
    return tuple(names)


class HeadBus(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def ping(self, dxl_id: int) -> int: ...
    def read1(self, dxl_id: int, address: int) -> int: ...
    def read4(self, dxl_id: int, address: int) -> int: ...
    def write1(self, dxl_id: int, address: int, value: int) -> None: ...
    def write2(self, dxl_id: int, address: int, value: int) -> None: ...
    def write4(self, dxl_id: int, address: int, value: int) -> None: ...


class SdkHeadBus:
    """Real serial bus through the ROBOTIS dynamixel_sdk (U2D2)."""

    def __init__(self, port: str, baud: int) -> None:
        self.port_name = port
        self.baud = int(baud)
        self._port: Any = None
        self._packet: Any = None

    def open(self) -> None:
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise HeadBusError(
                "dynamixel_sdk is missing: uv sync --extra head"
            ) from exc
        port = PortHandler(self.port_name)
        if not port.openPort():
            raise HeadBusError(f"cannot open {self.port_name}")
        if not port.setBaudRate(self.baud):
            port.closePort()
            raise HeadBusError(f"{self.port_name} rejected baud {self.baud}")
        self._port, self._packet = port, PacketHandler(2.0)

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None

    def _handles(self) -> tuple[Any, Any]:
        if self._port is None or self._packet is None:
            raise HeadBusError(f"{self.port_name} is not open")
        return self._port, self._packet

    def _check(self, comm: int, error: int, context: str) -> None:
        from dynamixel_sdk import COMM_SUCCESS

        _, packet = self._handles()
        if comm != COMM_SUCCESS:
            raise HeadBusError(f"{context}: {packet.getTxRxResult(comm)}")
        if error & 0x7F:  # bit 7 is the hardware-alert flag, read via ADDR 70
            raise HeadBusError(f"{context}: {packet.getRxPacketError(error)}")

    def ping(self, dxl_id: int) -> int:
        port, packet = self._handles()
        model, comm, error = packet.ping(port, dxl_id)
        self._check(comm, error, f"ping id={dxl_id}")
        return int(model)

    def read1(self, dxl_id: int, address: int) -> int:
        port, packet = self._handles()
        value, comm, error = packet.read1ByteTxRx(port, dxl_id, address)
        self._check(comm, error, f"read1 id={dxl_id} addr={address}")
        return int(value)

    def read4(self, dxl_id: int, address: int) -> int:
        port, packet = self._handles()
        value, comm, error = packet.read4ByteTxRx(port, dxl_id, address)
        self._check(comm, error, f"read4 id={dxl_id} addr={address}")
        value = int(value)
        return value - (1 << 32) if value & (1 << 31) else value

    def write1(self, dxl_id: int, address: int, value: int) -> None:
        port, packet = self._handles()
        comm, error = packet.write1ByteTxRx(port, dxl_id, address, int(value))
        self._check(comm, error, f"write1 id={dxl_id} addr={address}")

    def write2(self, dxl_id: int, address: int, value: int) -> None:
        port, packet = self._handles()
        comm, error = packet.write2ByteTxRx(port, dxl_id, address, int(value))
        self._check(comm, error, f"write2 id={dxl_id} addr={address}")

    def write4(self, dxl_id: int, address: int, value: int) -> None:
        port, packet = self._handles()
        comm, error = packet.write4ByteTxRx(port, dxl_id, address, int(value) & 0xFFFFFFFF)
        self._check(comm, error, f"write4 id={dxl_id} addr={address}")


@dataclass
class FakeHeadBus:
    """In-memory XC330 pair. Present position follows the goal while torque is on."""

    model: int = 1240
    present_ticks: dict[int, int] = field(default_factory=dict)
    hardware_error: dict[int, int] = field(default_factory=dict)
    fail_reads: int = 0  # raise on the next N reads (fault injection)
    writes: list[tuple[int, int, int]] = field(default_factory=list)
    is_open: bool = False
    _registers: dict[tuple[int, int], int] = field(default_factory=dict)

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def _require(self, dxl_id: int) -> None:
        if not self.is_open:
            raise HeadBusError("fake bus is closed")
        if dxl_id not in self.present_ticks:
            raise HeadBusError(f"no fake motor id={dxl_id}")

    def ping(self, dxl_id: int) -> int:
        self._require(dxl_id)
        return self.model

    def read1(self, dxl_id: int, address: int) -> int:
        self._require(dxl_id)
        if address == ADDR_HARDWARE_ERROR_STATUS:
            return self.hardware_error.get(dxl_id, 0)
        return self._registers.get((dxl_id, address), 0)

    def read4(self, dxl_id: int, address: int) -> int:
        self._require(dxl_id)
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise HeadBusError(f"fake read failure id={dxl_id}")
        if address == ADDR_PRESENT_POSITION:
            return self.present_ticks[dxl_id]
        return self._registers.get((dxl_id, address), 0)

    def _write(self, dxl_id: int, address: int, value: int) -> None:
        self._require(dxl_id)
        self.writes.append((dxl_id, address, int(value)))
        self._registers[(dxl_id, address)] = int(value)
        torque_on = self._registers.get((dxl_id, ADDR_TORQUE_ENABLE)) == TORQUE_ON
        if address == ADDR_GOAL_POSITION and torque_on:
            self.present_ticks[dxl_id] = int(value)

    write1 = _write
    write2 = _write
    write4 = _write


@dataclass(frozen=True)
class HeadHardwareConfig:
    pan_id: int
    tilt_id: int
    model_number: int = 1240
    operating_mode: int = 3
    position_i_gain: int = 400
    profile_acceleration: int = 20
    profile_velocity: int = 50
    start_tolerance_deg: float = 2.0


class HeadDriver:
    """Owns the pan/tilt motors; commands are clamped to the axis windows."""

    def __init__(
        self,
        bus: HeadBus,
        hardware: HeadHardwareConfig,
        *,
        pan_window: tuple[float, float],
        tilt_window: tuple[float, float],
    ) -> None:
        self.bus = bus
        self.hardware = hardware
        self.windows = {hardware.pan_id: pan_window, hardware.tilt_id: tilt_window}
        self.started = False

    @property
    def ids(self) -> tuple[int, int]:
        return (self.hardware.pan_id, self.hardware.tilt_id)

    def hardware_errors(self) -> dict[int, tuple[str, ...]]:
        errors = {}
        for dxl_id in self.ids:
            status = self.bus.read1(dxl_id, ADDR_HARDWARE_ERROR_STATUS)
            if status:
                errors[dxl_id] = decode_hardware_errors(status)
        return errors

    def read(self) -> tuple[float, float]:
        pan, tilt = (tick_to_deg(self.bus.read4(i, ADDR_PRESENT_POSITION)) for i in self.ids)
        return pan, tilt

    def start(self) -> tuple[float, float]:
        """Verify the motors and the start pose, then enable torque in place."""
        self.bus.open()
        try:
            for dxl_id in self.ids:
                model = self.bus.ping(dxl_id)
                if model != self.hardware.model_number:
                    raise HeadBusError(
                        f"id={dxl_id} model {model}, expected {self.hardware.model_number}"
                    )
            errors = self.hardware_errors()
            if errors:
                raise HeadBusError(f"hardware error latched: {errors}")
            measured = self.read()
            for dxl_id, value in zip(self.ids, measured, strict=True):
                lower, upper = self.windows[dxl_id]
                tol = self.hardware.start_tolerance_deg
                if not lower - tol <= value <= upper + tol:
                    raise HeadBusError(
                        f"id={dxl_id} starts at {value:.1f} deg, outside "
                        f"[{lower:.1f}, {upper:.1f}]; home the head first."
                    )
            for dxl_id in self.ids:
                self._configure(dxl_id)
        except Exception:
            self.bus.close()
            raise
        self.started = True
        return self.read()

    def _configure(self, dxl_id: int) -> None:
        hw = self.hardware
        self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        self.bus.write1(dxl_id, ADDR_OPERATING_MODE, hw.operating_mode)
        self.bus.write2(dxl_id, ADDR_POSITION_I_GAIN, hw.position_i_gain)
        self.bus.write4(dxl_id, ADDR_PROFILE_ACCELERATION, hw.profile_acceleration)
        self.bus.write4(dxl_id, ADDR_PROFILE_VELOCITY, hw.profile_velocity)
        present = self.bus.read4(dxl_id, ADDR_PRESENT_POSITION)
        self.bus.write4(dxl_id, ADDR_GOAL_POSITION, present)
        self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

    def command(self, pan_deg: float, tilt_deg: float) -> tuple[float, float]:
        if not self.started:
            raise HeadBusError("command() before start()")
        sent = []
        for dxl_id, value in zip(self.ids, (pan_deg, tilt_deg), strict=True):
            lower, upper = self.windows[dxl_id]
            clamped = min(max(float(value), lower), upper)
            self.bus.write4(dxl_id, ADDR_GOAL_POSITION, deg_to_tick(clamped))
            sent.append(clamped)
        return sent[0], sent[1]

    def stop(self, *, torque_off: bool = False) -> None:
        """Leave the head holding its last goal unless torque_off is requested.

        Torque off lets the tilt drop under the camera's weight, so it is opt-in.
        """
        try:
            if self.started and torque_off:
                for dxl_id in self.ids:
                    self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        finally:
            self.started = False
            self.bus.close()
