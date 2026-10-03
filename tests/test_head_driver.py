"""STEP 2 head driver (fake bus) and the head teleop loop."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from head_helpers import make_config, pose_from_yaw_pitch

from motion_acq.config import station_rig_config
from motion_acq.head.config import load_head_config
from motion_acq.head.dynamixel import (
    ADDR_GOAL_POSITION,
    ADDR_OPERATING_MODE,
    ADDR_POSITION_I_GAIN,
    ADDR_PROFILE_ACCELERATION,
    ADDR_PROFILE_VELOCITY,
    ADDR_TORQUE_ENABLE,
    FakeHeadBus,
    HeadBusError,
    HeadDriver,
    HeadHardwareConfig,
    decode_hardware_errors,
    deg_to_tick,
    tick_to_deg,
)
from motion_acq.head.retarget import HeadRetargeter, HeadState
from motion_acq.head.session import HeadSession

ROOT = Path(__file__).resolve().parents[1]
HW = HeadHardwareConfig(pan_id=1, tilt_id=2)
PAN_WINDOW = (-22.9, 17.1)
TILT_WINDOW = (56.8, 86.8)


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def fake_bus(pan: float = -2.9, tilt: float = 71.8) -> FakeHeadBus:
    return FakeHeadBus(present_ticks={1: deg_to_tick(pan), 2: deg_to_tick(tilt)})


def make_driver(bus: FakeHeadBus) -> HeadDriver:
    return HeadDriver(bus, HW, pan_window=PAN_WINDOW, tilt_window=TILT_WINDOW)


def test_tick_convention_matches_sim2real_home():
    # sim2real head_home_rh56f1.yaml: pan -2.9 deg <-> tick 2015, tilt 2865 <-> ~71.8.
    assert deg_to_tick(-2.9) == 2015
    assert tick_to_deg(2865) == pytest.approx(71.87, abs=0.01)
    assert tick_to_deg(deg_to_tick(12.34)) == pytest.approx(12.34, abs=0.09)


def test_hardware_error_decoding():
    assert decode_hardware_errors(0x24) == ("overheating", "overload")
    assert decode_hardware_errors(0x42) == ("unknown_0x42",)


def test_start_configures_in_sim2real_order_without_moving():
    bus = fake_bus()
    before = dict(bus.present_ticks)
    measured = make_driver(bus).start()
    assert bus.present_ticks == before
    assert measured == pytest.approx((tick_to_deg(before[1]), tick_to_deg(before[2])))
    for dxl_id in (1, 2):
        sequence = [(addr, value) for i, addr, value in bus.writes if i == dxl_id]
        assert [addr for addr, _ in sequence] == [
            ADDR_TORQUE_ENABLE, ADDR_OPERATING_MODE, ADDR_POSITION_I_GAIN,
            ADDR_PROFILE_ACCELERATION, ADDR_PROFILE_VELOCITY,
            ADDR_GOAL_POSITION, ADDR_TORQUE_ENABLE,
        ]
        assert sequence[0][1] == 0 and sequence[-1][1] == 1
        assert sequence[5][1] == before[dxl_id]  # goal seeded with present


def test_start_refuses_wrong_model():
    bus = fake_bus()
    bus.model = 1200
    with pytest.raises(HeadBusError, match="model"):
        make_driver(bus).start()
    assert not bus.is_open


def test_start_refuses_latched_hardware_error():
    bus = fake_bus()
    bus.hardware_error[2] = 0x20
    with pytest.raises(HeadBusError, match="overload"):
        make_driver(bus).start()
    assert not any(addr == ADDR_TORQUE_ENABLE for _, addr, _ in bus.writes)


def test_start_refuses_head_outside_window():
    bus = fake_bus(pan=40.0)
    with pytest.raises(HeadBusError, match="home the head first"):
        make_driver(bus).start()
    assert bus.writes == []


def test_command_is_clamped_to_window():
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    assert driver.command(100.0, 0.0) == (17.1, 56.8)
    assert bus.present_ticks[1] == deg_to_tick(17.1)


def test_stop_keeps_torque_unless_asked():
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    count = len(bus.writes)
    driver.stop()
    assert len(bus.writes) == count and not bus.is_open
    driver = make_driver(bus)
    driver.start()
    driver.stop(torque_off=True)
    assert bus.writes[-2:] == [(1, ADDR_TORQUE_ENABLE, 0), (2, ADDR_TORQUE_ENABLE, 0)]


def test_station_head_config_loads():
    config = load_head_config(station_rig_config("arm4090"), allow_fake_default=False)
    assert config.port.endswith("FT763P8T-if00-port0")
    assert config.pan_window == pytest.approx(PAN_WINDOW)
    assert config.tilt_window == pytest.approx(TILT_WINDOW)
    assert (config.hardware.pan_id, config.hardware.tilt_id) == (1, 2)


def test_station_without_head_refuses_real_but_allows_fake():
    with pytest.raises(SystemExit, match="no head configured"):
        load_head_config(station_rig_config("arm5080"), allow_fake_default=False)
    fake = load_head_config(station_rig_config("arm5080"), allow_fake_default=True)
    assert not fake.from_station


@dataclass
class FakeSample:
    device_hmd_pose: np.ndarray
    hmd_tracked: bool


@dataclass
class ScriptedTracker:
    samples: list[FakeSample]
    index: int = 0

    def latest(self) -> FakeSample:
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample


@dataclass
class Clock:
    t: float = 0.0
    dt: float = 0.02
    times: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        now = self.t
        self.t += self.dt
        return now


def run_session(samples, bus: FakeHeadBus, *, log=None, delay=0.1):
    driver = make_driver(bus)
    driver.start()
    session = HeadSession(
        ScriptedTracker(samples), driver,
        HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0)),
        auto_start_delay_s=delay, max_consecutive_faults=3, log_file=log, clock=Clock(),
    )
    steps = [session.tick() for _ in samples]
    return session, steps


def test_session_anchors_after_delay_then_follows():
    still = [FakeSample(pose_from_yaw_pitch(30, -20), True)] * 10
    turned = [FakeSample(pose_from_yaw_pitch(40, -20), True)] * 100
    bus = fake_bus()
    session, steps = run_session(still + turned, bus)
    assert steps[0] is None and session.anchored
    assert tick_to_deg(bus.present_ticks[1]) == pytest.approx(-2.9 + 10.0, abs=0.1)
    assert tick_to_deg(bus.present_ticks[2]) == pytest.approx(71.8, abs=0.1)


def test_session_never_commands_before_anchor_or_while_lost():
    lost = [FakeSample(pose_from_yaw_pitch(0, 0), False)] * 20
    bus = fake_bus()
    session, steps = run_session(lost, bus)
    assert not session.anchored and session.stats.commands == 0
    tracked = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 10
    moving = [FakeSample(pose_from_yaw_pitch(5, 0), True)] * 30
    gone = [FakeSample(pose_from_yaw_pitch(-30, 0), False)] * 30
    session, steps = run_session(tracked + moving + gone, bus)
    commands_before_loss = session.stats.commands
    assert all(s.state is HeadState.HOLD for s in steps[-30:])
    assert session.stats.holds == 30
    assert session.stats.commands == commands_before_loss


def test_session_stops_after_consecutive_faults():
    samples = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 50
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    session = HeadSession(
        ScriptedTracker(samples), driver, HeadRetargeter(make_config()),
        auto_start_delay_s=0.0, max_consecutive_faults=3, clock=Clock(),
    )
    session.tick()
    bus.fail_reads = 100
    with pytest.raises(HeadBusError, match="consecutive"):
        for _ in range(10):
            session.tick()


def test_session_log_has_step2_fields():
    samples = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 10
    log = io.StringIO()
    run_session(samples, fake_bus(), log=log, delay=0.0)
    rows = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(rows) == 10
    assert {
        "t_mono_s", "hmd_quat_xyzw", "hmd_tracked", "state", "rel_yaw_deg", "rel_pitch_deg",
        "filtered_yaw_deg", "filtered_pitch_deg", "cmd_pan_deg", "cmd_tilt_deg",
        "meas_pan_deg", "meas_tilt_deg",
    } <= rows[-1].keys()
    assert rows[-1]["state"] == "running"
