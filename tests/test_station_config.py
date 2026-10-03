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


@pytest.mark.parametrize("station", ["arm4090", "arm5080"])
def test_station_disables_j8_gripper(station):
    settings = load_openarm_settings(rig_config=station_rig_config(station))
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

    checks = collect_doctor_checks(station_rig_config("arm4090"), robot="openarmv1")
    cameras = [c for c in checks if c.name == "cameras"]
    assert cameras and cameras[0].status == "pass"


def test_doctor_still_fails_without_cameras_key(tmp_path) -> None:
    from motion_acq.scripts.doctor import collect_doctor_checks

    rig = tmp_path / "rig.yaml"
    rig.write_text("meta_quest: {}\n", encoding="utf-8")
    checks = collect_doctor_checks(rig, robot="openarmv1")
    assert [c.status for c in checks if c.name == "cameras"] == ["fail"]
