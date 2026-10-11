"""scripts/head_tilt_limit_probe.py: when the upward tilt sweep must stop."""

from __future__ import annotations

import importlib.util
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
