"""Head settings from the station rig (``head:`` section)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motion_acq.config import load_rig_config
from motion_acq.head.dynamixel import HeadHardwareConfig
from motion_acq.head.retarget import AxisConfig, RetargetConfig

# Used only with --backend fake on a station that has no head installed.
FAKE_HEAD_SECTION: dict[str, Any] = {
    "port": "",
    "baud": 1_000_000,
    "pan": {"id": 1, "home_deg": 0.0, "range_deg": 20.0},
    "tilt": {"id": 2, "home_deg": 0.0, "range_deg": 15.0},
}


@dataclass(frozen=True)
class HeadConfig:
    port: str
    baud: int
    hardware: HeadHardwareConfig
    retarget: RetargetConfig
    rate_hz: float = 50.0
    max_consecutive_faults: int = 10
    home_speed_deg_s: float = 20.0  # start and end: walk to home at this speed
    from_station: bool = True

    @property
    def home(self) -> tuple[float, float]:
        return (self.retarget.pan.home_deg, self.retarget.tilt.home_deg)

    @property
    def pan_window(self) -> tuple[float, float]:
        return (self.retarget.pan.lower_deg, self.retarget.pan.upper_deg)

    @property
    def tilt_window(self) -> tuple[float, float]:
        return (self.retarget.tilt.lower_deg, self.retarget.tilt.upper_deg)


def _axis(data: dict[str, Any], name: str) -> tuple[int, AxisConfig]:
    raw = data.get(name)
    if not isinstance(raw, dict):
        raise SystemExit(f"head.{name} must be a mapping with id, home_deg, range_deg.")
    try:
        dxl_id = int(raw["id"])
        axis = AxisConfig(
            home_deg=float(raw["home_deg"]),
            range_deg=float(raw["range_deg"]),
            sign=float(raw.get("sign", 1.0)),
            scale=float(raw.get("scale", 1.0)),
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid head.{name}: {exc}") from exc
    return dxl_id, axis


def _home_speed(limits: dict[str, Any]) -> float:
    speed = float(limits.get("home_speed_deg_s", 20.0))
    if not 0.0 < speed <= float(limits.get("max_velocity_deg_s", 60.0)):
        raise SystemExit(f"head.limits.home_speed_deg_s {speed} must be in (0, max_velocity_deg_s].")
    return speed


def head_config_from_section(data: dict[str, Any], *, from_station: bool = True) -> HeadConfig:
    pan_id, pan = _axis(data, "pan")
    tilt_id, tilt = _axis(data, "tilt")
    if pan_id == tilt_id:
        raise SystemExit("head.pan.id and head.tilt.id must differ.")
    motor = data.get("motor") or {}
    filt = data.get("filter") or {}
    limits = data.get("limits") or {}
    hardware = HeadHardwareConfig(
        pan_id=pan_id,
        tilt_id=tilt_id,
        model_number=int(motor.get("model_number", 1240)),
        operating_mode=int(motor.get("operating_mode", 3)),
        position_i_gain=int(motor.get("position_i_gain", 400)),
        profile_acceleration=int(motor.get("profile_acceleration", 20)),
        profile_velocity=int(motor.get("profile_velocity", 50)),
        start_tolerance_deg=float(motor.get("start_tolerance_deg", 2.0)),
    )
    retarget = RetargetConfig(
        pan=pan,
        tilt=tilt,
        deadband_deg=float(filt.get("deadband_deg", 0.5)),
        min_cutoff_hz=float(filt.get("min_cutoff_hz", 1.0)),
        beta=float(filt.get("beta", 0.05)),
        d_cutoff_hz=float(filt.get("d_cutoff_hz", 1.0)),
        max_velocity_deg_s=float(limits.get("max_velocity_deg_s", 60.0)),
        max_acceleration_deg_s2=float(limits.get("max_acceleration_deg_s2", 300.0)),
    )
    return HeadConfig(
        port=str(data.get("port") or ""),
        baud=int(data.get("baud", 1_000_000)),
        hardware=hardware,
        retarget=retarget,
        rate_hz=float(data.get("rate_hz", 50.0)),
        max_consecutive_faults=int(data.get("max_consecutive_faults", 10)),
        home_speed_deg_s=_home_speed(limits),
        from_station=from_station,
    )


def load_head_config(rig_config: Path, *, allow_fake_default: bool) -> HeadConfig:
    section = load_rig_config(rig_config).get("head")
    if isinstance(section, dict):
        return head_config_from_section(section)
    if allow_fake_default:
        return head_config_from_section(FAKE_HEAD_SECTION, from_station=False)
    raise SystemExit(f"No head section in {rig_config}; this station has no head configured.")
