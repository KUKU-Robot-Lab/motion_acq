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

import fcntl
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
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


class HeadAlertError(HeadBusError):
    """A motor raised its hardware-alert flag: stop commanding at once."""


class HeadPortBusyError(HeadBusError):
    """Another process holds the head serial port."""


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


def port_holders(device: str) -> list[int]:
    """PIDs (other than this one) with the serial device open, from /proc.

    Only processes of the same user are visible, which covers the s2r head
    nodes on the station PCs.
    """
    try:
        target = os.path.realpath(device)
    except OSError:
        return []
    holders = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                if os.path.realpath(fd) == target:
                    holders.append(int(entry.name))
                    break
        except OSError:
            continue
    return holders


class SdkHeadBus:
    """Real serial bus through the ROBOTIS dynamixel_sdk (U2D2).

    open() refuses a port that another process has open and takes an exclusive
    flock for its lifetime, so two motion_acq heads cannot share the bus.
    Serial/OS errors (USB unplugged) surface as HeadBusError; a response with
    the hardware-alert bit raises HeadAlertError except when reading the
    hardware error status itself.
    """

    ALERT_BIT = 0x80

    def __init__(self, port: str, baud: int, *, sdk: Any = None) -> None:
        self.port_name = port
        self.baud = int(baud)
        self._sdk = sdk
        self._port: Any = None
        self._packet: Any = None
        self._lock_fd: int | None = None

    def _sdk_module(self) -> Any:
        if self._sdk is None:
            try:
                import dynamixel_sdk
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise HeadBusError("dynamixel_sdk is missing: uv sync --extra head") from exc
            self._sdk = dynamixel_sdk
        return self._sdk

    def _lock(self) -> None:
        holders = port_holders(self.port_name)
        if holders:
            raise HeadPortBusyError(
                f"{self.port_name} is open in PID {holders}; stop that process first."
            )
        try:
            fd = os.open(self.port_name, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as exc:
            raise HeadBusError(f"cannot open {self.port_name}: {exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise HeadPortBusyError(f"{self.port_name} is locked by another process.") from exc
        self._lock_fd = fd

    def _unlock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def open(self) -> None:
        sdk = self._sdk_module()
        self._lock()
        try:
            port = sdk.PortHandler(self.port_name)
            if not port.openPort():
                raise HeadBusError(f"cannot open {self.port_name}")
            if not port.setBaudRate(self.baud):
                port.closePort()
                raise HeadBusError(f"{self.port_name} rejected baud {self.baud}")
        except HeadBusError:
            self._unlock()
            raise
        except (OSError, ValueError) as exc:
            self._unlock()
            raise HeadBusError(f"cannot open {self.port_name}: {exc}") from exc
        self._port, self._packet = port, sdk.PacketHandler(2.0)

    def close(self) -> None:
        try:
            if self._port is not None:
                self._port.closePort()
        except OSError:
            pass
        finally:
            self._port = None
            self._unlock()

    def _call(self, context: str, fn_name: str, *args: int, allow_alert: bool = False) -> Any:
        if self._port is None or self._packet is None:
            raise HeadBusError(f"{self.port_name} is not open")
        try:
            result = getattr(self._packet, fn_name)(self._port, *args)
        except (OSError, ValueError) as exc:  # serial.SerialException is an OSError
            raise HeadBusError(f"{context}: {exc}") from exc
        *value, comm, error = result
        if comm != self._sdk_module().COMM_SUCCESS:
            raise HeadBusError(f"{context}: {self._packet.getTxRxResult(comm)}")
        if error & self.ALERT_BIT and not allow_alert:
            raise HeadAlertError(f"{context}: hardware alert flag set")
        if error & ~self.ALERT_BIT:
            raise HeadBusError(f"{context}: {self._packet.getRxPacketError(error)}")
        return value[0] if value else None

    def ping(self, dxl_id: int) -> int:
        return int(self._call(f"ping id={dxl_id}", "ping", dxl_id, allow_alert=True))

    def read1(self, dxl_id: int, address: int) -> int:
        return int(self._call(
            f"read1 id={dxl_id} addr={address}", "read1ByteTxRx", dxl_id, address,
            allow_alert=address == ADDR_HARDWARE_ERROR_STATUS,
        ))

    def read4(self, dxl_id: int, address: int) -> int:
        value = int(self._call(f"read4 id={dxl_id} addr={address}", "read4ByteTxRx", dxl_id, address))
        return value - (1 << 32) if value & (1 << 31) else value

    def write1(self, dxl_id: int, address: int, value: int) -> None:
        self._call(f"write1 id={dxl_id} addr={address}", "write1ByteTxRx", dxl_id, address, int(value))

    def write2(self, dxl_id: int, address: int, value: int) -> None:
        self._call(f"write2 id={dxl_id} addr={address}", "write2ByteTxRx", dxl_id, address, int(value))

    def write4(self, dxl_id: int, address: int, value: int) -> None:
        self._call(
            f"write4 id={dxl_id} addr={address}", "write4ByteTxRx",
            dxl_id, address, int(value) & 0xFFFFFFFF,
        )


@dataclass
class FakeHeadBus:
    """In-memory XC330 pair. Present position follows the goal while torque is on."""

    model: int = 1240
    present_ticks: dict[int, int] = field(default_factory=dict)
    hardware_error: dict[int, int] = field(default_factory=dict)
    fail_reads: int = 0  # raise on the next N reads (fault injection)
    fail_writes: int = 0  # raise on the next N writes
    alert: bool = False  # every exchange reports the hardware-alert flag
    # Ticks a motor drops when its torque is switched off (tilt under the camera).
    sag_ticks_on_torque_off: dict[int, int] = field(default_factory=dict)
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
        if self.alert:
            raise HeadAlertError(f"fake alert id={dxl_id}")
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise HeadBusError(f"fake read failure id={dxl_id}")
        if address == ADDR_PRESENT_POSITION:
            return self.present_ticks[dxl_id]
        return self._registers.get((dxl_id, address), 0)

    def _write(self, dxl_id: int, address: int, value: int) -> None:
        self._require(dxl_id)
        if self.alert:
            raise HeadAlertError(f"fake alert id={dxl_id}")
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise HeadBusError(f"fake write failure id={dxl_id} addr={address}")
        was_on = self._registers.get((dxl_id, ADDR_TORQUE_ENABLE)) == TORQUE_ON
        self.writes.append((dxl_id, address, int(value)))
        self._registers[(dxl_id, address)] = int(value)
        torque_on = self._registers.get((dxl_id, ADDR_TORQUE_ENABLE)) == TORQUE_ON
        if address == ADDR_TORQUE_ENABLE and value == TORQUE_OFF:
            self.present_ticks[dxl_id] += self.sag_ticks_on_torque_off.get(dxl_id, 0)
        if address == ADDR_TORQUE_ENABLE and torque_on and not was_on:
            goal = self._registers.get((dxl_id, ADDR_GOAL_POSITION))
            if goal is not None:
                self.present_ticks[dxl_id] = goal
        if address == ADDR_GOAL_POSITION and torque_on:
            self.present_ticks[dxl_id] = int(value)

    def torque_on(self, dxl_id: int) -> bool:
        return self._registers.get((dxl_id, ADDR_TORQUE_ENABLE)) == TORQUE_ON

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

    def _check_window(self, dxl_id: int, value: float) -> None:
        lower, upper = self.windows[dxl_id]
        tol = self.hardware.start_tolerance_deg
        if not lower - tol <= value <= upper + tol:
            raise HeadBusError(
                f"id={dxl_id} is at {value:.1f} deg, outside "
                f"[{lower:.1f}, {upper:.1f}]; home the head first."
            )

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
                self._check_window(dxl_id, value)
            for dxl_id in self.ids:
                self._configure(dxl_id)
        except Exception:
            self.bus.close()
            raise
        self.started = True
        return self.read()

    def _configure(self, dxl_id: int) -> None:
        """Mode/gain/profile need torque off; any failure re-enables torque in place."""
        hw = self.hardware
        self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
        try:
            self.bus.write1(dxl_id, ADDR_OPERATING_MODE, hw.operating_mode)
            self.bus.write2(dxl_id, ADDR_POSITION_I_GAIN, hw.position_i_gain)
            self.bus.write4(dxl_id, ADDR_PROFILE_ACCELERATION, hw.profile_acceleration)
            self.bus.write4(dxl_id, ADDR_PROFILE_VELOCITY, hw.profile_velocity)
            present = self.bus.read4(dxl_id, ADDR_PRESENT_POSITION)
            self._check_window(dxl_id, tick_to_deg(present))  # sag after torque off
            self.bus.write4(dxl_id, ADDR_GOAL_POSITION, present)
            self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        except Exception:
            self._restore_hold(dxl_id)
            raise

    def _restore_hold(self, dxl_id: int) -> None:
        """Best effort: hold where the motor is now rather than leave it limp."""
        try:
            present = self.bus.read4(dxl_id, ADDR_PRESENT_POSITION)
            self.bus.write4(dxl_id, ADDR_GOAL_POSITION, present)
            self.bus.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        except HeadBusError:
            pass

    def command(self, pan_deg: float, tilt_deg: float) -> tuple[float, float]:
        if not self.started:
            raise HeadBusError("command() before start()")
        if not (math.isfinite(pan_deg) and math.isfinite(tilt_deg)):
            raise HeadBusError(f"non-finite head command ({pan_deg}, {tilt_deg})")
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
