"""Station rig selection and the gripper-less OpenArm path (STEP 1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motion_acq.config import station_rig_config
from motion_acq.real.openarm.driver import OpenArmSdkSide, load_openarm_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_station_falls_back_to_local_rig_when_unset(monkeypatch):
    monkeypatch.delenv("MACQ_STATION", raising=False)
    assert station_rig_config() == Path("configs/rig.yaml")


@pytest.mark.parametrize("station", ["arm4090", "arm5080"])
def test_station_resolves_to_its_own_file(station):
    assert station_rig_config(station) == Path(f"configs/stations/{station}.yaml")


def test_unknown_station_is_refused():
    with pytest.raises(SystemExit, match="Unknown station"):
        station_rig_config("nope")


STATION_ROBOT = {"arm4090": "openarm_rh56f1", "arm5080": "openarmv1"}


@pytest.mark.parametrize("station", ["arm4090", "arm5080"])
def test_station_disables_j8_gripper(station):
    settings = load_openarm_settings(
        rig_config=station_rig_config(station), robot_name=STATION_ROBOT[station]
    )
    assert settings.gripper_enabled is False


def test_gripper_enabled_by_default_without_rig(tmp_path):
    settings = load_openarm_settings(rig_config=tmp_path / "missing.yaml")
    assert settings.gripper_enabled is True


class _FakeArm:
    def __init__(self):
        self.calls: list[str] = []

    def init_arm_motors(self, *_):
        self.calls.append("init_arm")

    def init_gripper_motor(self, *_):
        self.calls.append("init_gripper")

    def set_callback_mode_all(self, *_):
        pass

    def enable_all(self):
        self.calls.append("enable_all")

    def recv_all(self, *_):
        pass

    def get_arm(self):
        return SimpleNamespace(mit_control_all=lambda params: self.calls.append("mit"))

    def get_gripper(self):
        return SimpleNamespace(set_position=lambda pos: self.calls.append("gripper"))


def _fake_sdk(arm: _FakeArm):
    motor = SimpleNamespace(DM8009=0, DM4340=1, DM4310=2)
    return SimpleNamespace(
        MotorType=motor,
        OpenArm=lambda port, fd: arm,
        ControlMode=SimpleNamespace(POS_FORCE=0),
        CallbackMode=SimpleNamespace(STATE=0),
        MITParam=lambda *a: a,
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_gripper_motor_is_only_touched_when_enabled(enabled):
    arm = _FakeArm()
    side = OpenArmSdkSide(
        "can0",
        enable_fd=True,
        kp=(1.0,) * 7,
        kd=(0.1,) * 7,
        gripper_enabled=enabled,
        sdk=_fake_sdk(arm),
    )
    side.send(np.zeros(7, dtype=np.float32), 0.5)
    assert ("init_gripper" in arm.calls) is enabled
    assert ("gripper" in arm.calls) is enabled
    assert "mit" in arm.calls


def test_doctor_accepts_explicit_empty_cameras() -> None:
    from motion_acq.scripts.doctor import collect_doctor_checks

    checks = collect_doctor_checks(station_rig_config("arm5080"), robot="openarmv1")
    cameras = [c for c in checks if c.name == "cameras"]
    assert cameras and cameras[0].status == "pass"


def test_doctor_checks_the_quest_view_head_camera_stream() -> None:
    from motion_acq.scripts.doctor import collect_doctor_checks

    checks = collect_doctor_checks(station_rig_config("arm4090"), robot="openarm_rh56f1")
    head = [c for c in checks if c.name == "camera:head"]
    assert head and "quest-view frames 127.0.0.1:47126" in head[0].detail


def test_doctor_still_fails_without_cameras_key(tmp_path) -> None:
    from motion_acq.scripts.doctor import collect_doctor_checks

    rig = tmp_path / "rig.yaml"
    rig.write_text("meta_quest: {}\n", encoding="utf-8")
    checks = collect_doctor_checks(rig, robot="openarmv1")
    assert [c.status for c in checks if c.name == "cameras"] == ["fail"]


@pytest.mark.parametrize(
    ("station", "left", "right", "auto_repair"),
    [("arm4090", "can1", "can0", False), ("arm5080", "can1", "can0", True)],
)
def test_station_can_ports(station, left, right, auto_repair):
    settings = load_openarm_settings(
        rig_config=station_rig_config(station), robot_name=STATION_ROBOT[station]
    )
    assert (settings.left_port, settings.right_port) == (left, right)
    assert settings.can_auto_repair is auto_repair
    assert settings.enable_fd and settings.bitrate == 1_000_000
    assert settings.dbitrate == 5_000_000


@pytest.mark.parametrize(("auto_repair", "requested", "expected"), [
    (False, True, False), (True, True, True), (True, False, False),
])
def test_prepare_never_repairs_when_station_forbids_it(
    monkeypatch, auto_repair, requested, expected
):
    from motion_acq.real import can_setup
    from motion_acq.real.openarm.driver import OpenArmCanEnvironment, OpenArmCanSettings

    seen = {}

    def fake_ready(ports, *, bitrate, dbitrate, repair):
        seen["ports"], seen["repair"] = list(ports), repair
        return {}

    monkeypatch.setattr(can_setup, "ensure_can_fd_interfaces_ready", fake_ready)
    env = OpenArmCanEnvironment(
        OpenArmCanSettings(left_port="can3", right_port="can2", can_auto_repair=auto_repair),
        active_sides=("left", "right"),
        joint_limits={},
    )
    env.prepare(repair=requested)
    assert seen == {"ports": ["can3", "can2"], "repair": expected}


def test_glove_driver_is_not_a_can_holder():
    from motion_acq.scripts.station import can_holders

    out = (
        "4242 /opt/ros/humble/lib/controller_manager/ros2_control_node --ros-args "
        "-r __ns:=/senseglove/glove00782/rh --params-file /tmp/x.yaml\n"
        "17864 /usr/bin/python3 /opt/ros/humble/bin/ros2 launch openarm_bringup openarm.bimanual.launch.py\n"
    )
    assert can_holders(out) == [out.splitlines()[1]]
    assert can_holders("") == []


def test_hand_calibration_from_before_sensecom_is_kept_but_flagged(tmp_path):
    """10.05 user: the calibration looked lost after every glove reconnect."""
    import os

    from motion_acq.scripts.station import _process_start_epoch, calibration_current, calibration_stale

    cal = tmp_path / "op1_right.yaml"
    assert not calibration_current(cal, None)[0]
    cal.write_text("x")
    os.utime(cal, (1000.0, 1000.0))
    assert calibration_current(cal, None)[0]
    assert calibration_current(cal, 999.0)[0]
    ok, detail = calibration_current(cal, 1001.0)
    assert ok and "recalibrate" in detail
    assert calibration_stale(cal, 1001.0) and not calibration_stale(cal, 999.0)
    assert not calibration_stale(cal, None)
    import time
    start = _process_start_epoch(os.getpid())
    assert start is not None and start <= time.time()
