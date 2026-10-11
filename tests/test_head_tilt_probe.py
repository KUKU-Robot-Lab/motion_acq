"""scripts/head_tilt_limit_probe.py: when the upward tilt sweep must stop."""

from __future__ import annotations

import importlib.util

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "head_tilt_limit_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)  # type: ignore[union-attr]

LIMITS = {"max_lag_deg": 3.0, "current_margin_ma": 150.0}


def test_free_motion_continues():
    assert probe.contact(40.0, 40.4, 120.0, 80.0, **LIMITS) is None


def test_a_blocked_tilt_stops_on_lag():
    assert "lag" in probe.contact(30.0, 33.5, 90.0, 80.0, **LIMITS)


def test_pushing_on_a_stop_stops_on_current():
    assert "current" in probe.contact(30.0, 30.2, -260.0, 80.0, **LIMITS)


def test_present_current_is_signed():
    assert probe.signed16(0x0000FF9C) == -100
    assert probe.signed16(0x12340064) == 100


def test_camera_pitch_from_the_raw_accel():
    assert probe.camera_pitch_deg(32, -305, 950) == pytest.approx(-72.1, abs=0.1)
    assert probe.camera_pitch_deg(0, -980, 0) == pytest.approx(0.0, abs=1e-9)
    assert probe.camera_pitch_deg(0, -500, -500) == pytest.approx(45.0)


def _fake_iio(tmp_path, name, xyz):
    dev = tmp_path / "iio:device3"
    dev.mkdir()
    (dev / "name").write_text(name + "\n")
    for a, v in zip("xyz", xyz):
        (dev / f"in_accel_{a}_raw").write_text(f"{v}\n")
    return dev


def test_the_accel_is_found_by_name_and_read(tmp_path):
    (tmp_path / "iio:device0").mkdir()
    (tmp_path / "iio:device0" / "name").write_text("gyro_3d\n")
    dev = _fake_iio(tmp_path, "accel_3d", (0, -980, 0))
    assert probe.find_accel(tmp_path) == dev
    assert probe.read_pitch(dev) == pytest.approx(0.0, abs=1e-9)


def test_no_accel_leaves_the_pitch_empty(tmp_path):
    assert probe.find_accel(tmp_path) is None
    assert probe.read_pitch(None) is None
    assert probe.read_pitch(tmp_path / "gone") is None
