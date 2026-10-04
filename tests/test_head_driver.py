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
    HeadAlertError,
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
    assert config.tilt_window == pytest.approx((41.8, 86.8))  # 30 deg up (encoder -), 15 deg down
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


def make_session(samples, bus: FakeHeadBus, *, delay=0.0, faults=3) -> HeadSession:
    driver = make_driver(bus)
    driver.start()
    return HeadSession(
        ScriptedTracker(samples), driver,
        HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0)),
        auto_start_delay_s=delay, max_consecutive_faults=faults, clock=Clock(),
    )


def goal_writes(bus: FakeHeadBus) -> int:
    return sum(1 for _, addr, _ in bus.writes if addr == ADDR_GOAL_POSITION)


def test_session_never_commands_before_anchor():
    lost = [FakeSample(pose_from_yaw_pitch(0, 0), False)] * 20
    bus = fake_bus()
    session = make_session(lost, bus)
    writes = goal_writes(bus)
    for _ in lost:
        session.tick()
    assert not session.anchored and session.stats.commands == 0
    assert goal_writes(bus) == writes


def test_session_writes_no_goal_while_hmd_lost():
    tracked = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 10
    moving = [FakeSample(pose_from_yaw_pitch(5, 0), True)] * 30
    gone = [FakeSample(pose_from_yaw_pitch(-30, 0), False)] * 30
    bus = fake_bus()
    session = make_session(tracked + moving + gone, bus)
    for _ in tracked + moving:
        session.tick()
    commands, writes, present = session.stats.commands, goal_writes(bus), dict(bus.present_ticks)
    assert commands > 0
    steps = [session.tick() for _ in gone]
    assert all(s.state is HeadState.HOLD for s in steps)
    assert session.stats.holds == 30
    assert session.stats.commands == commands
    assert goal_writes(bus) == writes and bus.present_ticks == present


def test_session_stops_when_reads_fail_even_if_writes_succeed():
    samples = [FakeSample(pose_from_yaw_pitch(k / 5, 0), True) for k in range(200)]
    bus = fake_bus()
    session = make_session(samples, bus, faults=3)
    session.tick()
    bus.fail_reads = 1000
    with pytest.raises(HeadBusError, match="consecutive faulty cycles"):
        for _ in range(10):
            session.tick()
    assert session.stats.commands > 0  # the writes kept succeeding


def test_session_stops_at_once_on_hardware_alert():
    samples = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 20
    bus = fake_bus()
    session = make_session(samples, bus, faults=100)
    session.tick()
    bus.alert = True
    with pytest.raises(HeadAlertError):
        session.tick()


def test_session_stops_on_runtime_hardware_error():
    samples = [FakeSample(pose_from_yaw_pitch(0, 0), True)] * 200
    bus = fake_bus()
    session = make_session(samples, bus)
    session.tick()
    bus.hardware_error[1] = 0x20
    with pytest.raises(HeadBusError, match="overload"):
        for _ in range(100):  # checked once per second (50 ticks at 50 Hz)
            session.tick()


def test_non_finite_command_is_refused_without_writing():
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    writes = len(bus.writes)
    with pytest.raises(HeadBusError, match="non-finite"):
        driver.command(float("nan"), 70.0)
    assert len(bus.writes) == writes


def test_configure_failure_restores_torque_in_place():
    bus = fake_bus()
    bus.fail_writes = 0
    driver = make_driver(bus)
    original = bus._write

    def fail_on_mode(dxl_id, address, value):
        if address == ADDR_OPERATING_MODE and dxl_id == 2:
            raise HeadBusError("fake mode write failure")
        original(dxl_id, address, value)

    bus.write1 = fail_on_mode  # type: ignore[method-assign]
    with pytest.raises(HeadBusError, match="mode write"):
        driver.start()
    assert bus.torque_on(1) and bus.torque_on(2)
    assert not bus.is_open


def test_sag_out_of_window_after_torque_off_aborts_but_holds():
    bus = fake_bus(tilt=57.5)  # 0.7 deg inside the lower tilt window edge
    bus.sag_ticks_on_torque_off[2] = -40  # ~3.5 deg drop once torque is off
    with pytest.raises(HeadBusError, match="home the head first"):
        make_driver(bus).start()
    assert bus.torque_on(2)


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


# -- start and end pose = home ---------------------------------------------------

class FakeTime:
    def __init__(self) -> None:
        self.t = 0.0

    def sleep(self, dt: float) -> None:
        self.t += dt

    def clock(self) -> float:
        return self.t


def goal_trace(bus: FakeHeadBus, dxl_id: int) -> list[float]:
    return [tick_to_deg(v) for i, a, v in bus.writes if i == dxl_id and a == ADDR_GOAL_POSITION]


def test_move_to_walks_home_at_the_speed_and_waits_for_arrival():
    bus = fake_bus(pan=7.1, tilt=61.8)  # 10 deg off home on both axes, inside the windows
    driver = make_driver(bus)
    driver.start()
    clock = FakeTime()
    start_writes = len(bus.writes)
    measured = driver.move_to(-2.9, 71.8, speed_deg_s=20.0, rate_hz=50.0,
                              sleep=clock.sleep, clock=clock.clock)
    assert measured == pytest.approx((-2.9, 71.8), abs=0.1)
    pan = [tick_to_deg(v) for i, a, v in bus.writes[start_writes:] if i == 1 and a == ADDR_GOAL_POSITION]
    steps = [abs(b - a) for a, b in zip([7.1] + pan, pan, strict=False)]
    # 10 deg (plus tick rounding) at 0.4 deg per 20 ms
    assert len(pan) in (25, 26) and max(steps) <= 20.0 / 50.0 + 0.1
    assert clock.t == pytest.approx(0.52, abs=0.05)


def test_move_to_raises_if_the_head_does_not_arrive():
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    driver.bus.write4 = lambda *a: None  # goals lost: the head never moves
    clock = FakeTime()
    with pytest.raises(HeadBusError, match="did not reach"):
        driver.move_to(7.1, 71.8, speed_deg_s=20.0, sleep=clock.sleep, clock=clock.clock)


def test_head_teleop_starts_and_ends_at_home(monkeypatch):
    """Fake backend run: walk home, follow briefly, walk home again on exit."""
    from motion_acq.scripts import head_teleop

    bus = fake_bus(pan=7.1, tilt=61.8)
    monkeypatch.setattr(head_teleop, "build_bus", lambda args, config: bus)

    class NoQuest:
        def start(self): ...
        def stop(self): ...
        def latest(self):
            from motion_acq.tracking.base import ControllerPairSample
            return ControllerPairSample.empty() if hasattr(ControllerPairSample, "empty") else None

    monkeypatch.setattr(head_teleop, "build_tracker", lambda args: NoQuest())
    monkeypatch.setattr(head_teleop, "_run", lambda session, rate, duration: bus.writes.append(("run",)))
    head_teleop.main(["--backend", "fake", "--no-log", "--rig-config",
                      str(station_rig_config("arm4090"))])
    run_at = bus.writes.index(("run",))
    before = [w for w in bus.writes[:run_at] if w[0] == 1 and w[1] == ADDR_GOAL_POSITION]
    after = [w for w in bus.writes[run_at + 1:] if w[0] == 1 and w[1] == ADDR_GOAL_POSITION]
    assert tick_to_deg(before[-1][2]) == pytest.approx(-2.9, abs=0.1)
    assert bus.present_ticks[1] == deg_to_tick(-2.9) and bus.present_ticks[2] == deg_to_tick(71.8)
    assert after and tick_to_deg(after[-1][2]) == pytest.approx(-2.9, abs=0.1)


def test_home_tolerance_covers_the_arm4090_pan_static_error():
    """s2r: pan goal 1997 ticks rests at 2015 (1.6 deg); 10.04 run stopped 1.5 deg short of home."""
    config = load_head_config(station_rig_config("arm4090"), allow_fake_default=False)
    assert config.home_tolerance_deg >= 1.6
    bus = fake_bus(pan=-1.4, tilt=71.9)
    driver = make_driver(bus)
    driver.start()
    driver.bus.write4 = lambda *a: None  # the motor does not move for a 1.5 deg goal change
    clock = FakeTime()
    measured = driver.move_to(*config.home, speed_deg_s=20.0, tolerance_deg=config.home_tolerance_deg,
                              sleep=clock.sleep, clock=clock.clock)
    assert measured == pytest.approx((-1.4, 71.9), abs=0.1)


def test_direction_check_reads_the_image_shift():
    import importlib.util

    import cv2

    spec = importlib.util.spec_from_file_location("hdc", ROOT / "scripts" / "head_direction_check.py")
    hdc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdc)
    rng = np.random.default_rng(0)
    base = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 1)
    # 10.04: auto exposure brightened the moved frame and fooled phase correlation
    right = cv2.convertScaleAbs(np.roll(base, 12, axis=1), alpha=1.6, beta=30)  # scene right: camera left
    up = cv2.convertScaleAbs(np.roll(base, -10, axis=0), alpha=0.6, beta=-10)  # scene up: camera down
    dx, _ = hdc.image_shift(base, right)
    _, dy = hdc.image_shift(base, up)
    assert dx == pytest.approx(12, abs=1) and dy == pytest.approx(-10, abs=1)
    assert hdc.sign_from_shift("pan", 5, (dx, 0.0))[0] == 1
    assert hdc.sign_from_shift("tilt", 5, (0.0, dy))[0] == -1
    assert hdc.sign_from_shift("pan", 5, (0.5, 0.0))[0] is None


def test_key_unlock_follows_from_the_current_direction_and_locks_again():
    """10.04 user: the operator cannot match the home pose, so the head waits locked
    at home; Space anchors at wherever they look, Space again returns it to home."""
    bus = fake_bus()
    driver = make_driver(bus)
    driver.start()
    # each block has one extra sample: the tick right after a toggle anchors on it
    poses = iter([pose_from_yaw_pitch(35, -15)] * 31 + [pose_from_yaw_pitch(45, -15)] * 59
                 + [pose_from_yaw_pitch(10, -15)] * 31 + [pose_from_yaw_pitch(20, -15)] * 59)

    class Tracker:
        def latest(self):
            return FakeSample(next(poses), True)

    session = HeadSession(
        Tracker(), driver,
        HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0)),
        auto_start_delay_s=2.0, max_consecutive_faults=3, clock=Clock(), unlock="key",
    )
    writes = goal_writes(bus)
    for _ in range(30):  # HMD tracked and turned 35 deg away: still locked, nothing sent
        session.tick()
    assert session.locked and goal_writes(bus) == writes
    session.toggle_lock()  # unlock while looking 35 deg left: that direction is the anchor
    session.tick()
    for _ in range(59):
        session.tick()
    assert not session.locked
    pan = tick_to_deg(bus.present_ticks[1])
    assert pan == pytest.approx(-2.9 + 10.0, abs=0.2)  # followed the +10 deg turn only
    session.toggle_lock()
    for _ in range(30):  # locked: back to home at 20 deg/s (10 deg in 0.5 s), whatever the HMD does
        session.tick()
    assert not session.returning
    assert tick_to_deg(bus.present_ticks[1]) == pytest.approx(-2.9, abs=0.1)
    session.toggle_lock()  # unlock again: anchors at home and this HMD direction, no jump
    session.tick()
    assert tick_to_deg(bus.present_ticks[1]) == pytest.approx(-2.9, abs=0.2)
    for _ in range(59):
        session.tick()
    assert tick_to_deg(bus.present_ticks[1]) == pytest.approx(-2.9 + 10.0, abs=0.3)
