"""SdkHeadBus against a fake dynamixel_sdk: decoding, errors and port locking."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from motion_acq.head import dynamixel as dx
from motion_acq.head.dynamixel import (
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_PRESENT_POSITION,
    HeadAlertError,
    HeadBusError,
    HeadPortBusyError,
    SdkHeadBus,
)

COMM_SUCCESS = 0
COMM_TX_FAIL = -1001


class FakePort:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def openPort(self) -> bool:  # noqa: N802 - dynamixel_sdk API
        return True

    def setBaudRate(self, baud: int) -> bool:  # noqa: N802
        return baud == 1_000_000

    def closePort(self) -> None:  # noqa: N802
        self.closed = True


class FakePacket:
    def __init__(self) -> None:
        self.reply: dict[str, tuple] = {}
        self.calls: list[tuple] = []
        self.raise_exc: Exception | None = None

    def _answer(self, name: str, *args):
        self.calls.append((name, *args))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.reply[name]

    def ping(self, port, dxl_id):
        return self._answer("ping", dxl_id)

    def read1ByteTxRx(self, port, dxl_id, addr):  # noqa: N802
        return self._answer("read1", dxl_id, addr)

    def read4ByteTxRx(self, port, dxl_id, addr):  # noqa: N802
        return self._answer("read4", dxl_id, addr)

    def write1ByteTxRx(self, port, dxl_id, addr, value):  # noqa: N802
        return self._answer("write1", dxl_id, addr, value)

    def write2ByteTxRx(self, port, dxl_id, addr, value):  # noqa: N802
        return self._answer("write2", dxl_id, addr, value)

    def write4ByteTxRx(self, port, dxl_id, addr, value):  # noqa: N802
        return self._answer("write4", dxl_id, addr, value)

    def getTxRxResult(self, comm):  # noqa: N802
        return f"comm {comm}"

    def getRxPacketError(self, error):  # noqa: N802
        return f"packet error 0x{error:02x}"


@pytest.fixture
def device(tmp_path):
    path = tmp_path / "ttyFAKE"
    path.write_bytes(b"")
    return str(path)


@pytest.fixture
def packet():
    return FakePacket()


@pytest.fixture
def bus(device, packet, monkeypatch):
    monkeypatch.setattr(dx, "port_holders", lambda _device: [])
    sdk = SimpleNamespace(
        PortHandler=FakePort, PacketHandler=lambda protocol: packet, COMM_SUCCESS=COMM_SUCCESS
    )
    bus = SdkHeadBus(device, 1_000_000, sdk=sdk)
    bus.open()
    yield bus
    bus.close()


def test_read4_is_signed(bus, packet):
    packet.reply["read4"] = (0xFFFFFFFE, COMM_SUCCESS, 0)
    assert bus.read4(1, ADDR_PRESENT_POSITION) == -2
    packet.reply["read4"] = (2865, COMM_SUCCESS, 0)
    assert bus.read4(1, ADDR_PRESENT_POSITION) == 2865


def test_write4_masks_negative_to_uint32(bus, packet):
    packet.reply["write4"] = (COMM_SUCCESS, 0)
    bus.write4(1, 116, -2)
    assert packet.calls[-1] == ("write4", 1, 116, 0xFFFFFFFE)


def test_comm_failure_raises(bus, packet):
    packet.reply["read4"] = (0, COMM_TX_FAIL, 0)
    with pytest.raises(HeadBusError, match="comm -1001"):
        bus.read4(1, ADDR_PRESENT_POSITION)


def test_packet_error_raises_and_alert_bit_is_separate(bus, packet):
    packet.reply["write1"] = (COMM_SUCCESS, 0x02)
    with pytest.raises(HeadBusError, match="packet error 0x02") as info:
        bus.write1(1, 64, 1)
    assert not isinstance(info.value, HeadAlertError)
    packet.reply["write1"] = (COMM_SUCCESS, 0x80)
    with pytest.raises(HeadAlertError):
        bus.write1(1, 64, 1)


def test_alert_allowed_when_reading_hardware_status_and_ping(bus, packet):
    packet.reply["read1"] = (0x20, COMM_SUCCESS, 0x80)
    assert bus.read1(1, ADDR_HARDWARE_ERROR_STATUS) == 0x20
    packet.reply["ping"] = (1240, COMM_SUCCESS, 0x80)
    assert bus.ping(1) == 1240
    with pytest.raises(HeadAlertError):
        bus.read1(1, 146)


def test_serial_exception_becomes_bus_error(bus, packet):
    packet.raise_exc = OSError(5, "Input/output error")  # USB unplugged
    with pytest.raises(HeadBusError, match="Input/output"):
        bus.read4(1, ADDR_PRESENT_POSITION)


def test_second_open_on_same_port_is_refused(bus, device, packet, monkeypatch):
    sdk = SimpleNamespace(
        PortHandler=FakePort, PacketHandler=lambda protocol: packet, COMM_SUCCESS=COMM_SUCCESS
    )
    other = SdkHeadBus(device, 1_000_000, sdk=sdk)
    with pytest.raises(HeadPortBusyError, match="locked"):
        other.open()
    bus.close()
    other.open()  # the lock is released on close
    other.close()


def test_port_held_by_another_process_is_refused(device, packet, monkeypatch):
    monkeypatch.setattr(dx, "port_holders", lambda _device: [4242])
    sdk = SimpleNamespace(
        PortHandler=FakePort, PacketHandler=lambda protocol: packet, COMM_SUCCESS=COMM_SUCCESS
    )
    with pytest.raises(HeadPortBusyError, match="4242"):
        SdkHeadBus(device, 1_000_000, sdk=sdk).open()


def test_port_holders_finds_an_open_file(device):
    # This process is excluded; another process holding the file must be found.
    holder = subprocess.Popen(
        [sys.executable, "-c", f"f = open({device!r}); print('ready', flush=True); input()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
        assert holder.pid in dx.port_holders(device)
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.pid not in dx.port_holders(device)


def test_rejected_baud_releases_the_lock(device, packet, monkeypatch):
    monkeypatch.setattr(dx, "port_holders", lambda _device: [])
    sdk = SimpleNamespace(
        PortHandler=FakePort, PacketHandler=lambda protocol: packet, COMM_SUCCESS=COMM_SUCCESS
    )
    with pytest.raises(HeadBusError, match="rejected baud"):
        SdkHeadBus(device, 57_600, sdk=sdk).open()
    good = SdkHeadBus(device, 1_000_000, sdk=sdk)
    good.open()
    good.close()
