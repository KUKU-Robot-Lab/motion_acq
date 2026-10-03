"""Shared loading for the machine-local HandUMI rig configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

STATIONS_DIR = Path("configs/stations")
STATION_ENV = "MACQ_STATION"


def station_rig_config(station: str | None = None) -> Path:
    """Rig file for a station (arm4090, arm5080); falls back to configs/rig.yaml.

    The station comes from the argument or the ``MACQ_STATION`` environment
    variable, so each robot PC selects its own CAN ports, Quest IP and cameras.
    """
    name = (station if station is not None else os.environ.get(STATION_ENV, "")).strip()
    if not name:
        return Path("configs/rig.yaml")
    path = STATIONS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(item.stem for item in STATIONS_DIR.glob("*.yaml"))
        raise SystemExit(
            f"Unknown station {name!r}: {path} is missing. Available: {available or 'none'}."
        )
    return path


DEFAULT_RIG_CONFIG = station_rig_config()


def station_default_robot(fallback: str = "openarmv1") -> str:
    """recording.robot of the selected station rig (arm4090: openarm_rh56f1)."""
    try:
        with DEFAULT_RIG_CONFIG.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError:
        return fallback
    return str((data.get("recording") or {}).get("robot") or fallback)
SIDES = ("left", "right")


def resolve_active_sides(
    side: str | None = None, *, available: tuple[str, ...] = SIDES
) -> tuple[str, ...]:
    """Select existing arms; omission follows the embodiment's topology."""
    selected = available if side is None else SIDES if side == "both" else (side,)
    if not selected or any(value not in available for value in selected):
        raise ValueError(f"Side {side!r} is unavailable; this embodiment has {available}.")
    return selected


def dataset_active_sides(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Legacy captures are bilateral; new captures declare their active sensors."""
    values = metadata.get("active_sides", SIDES)
    if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
        raise ValueError("handumi.active_sides must contain left, right, or both.")
    if any(side not in SIDES for side in values):
        raise ValueError("handumi.active_sides must contain left, right, or both.")
    return tuple(side for side in SIDES if side in values)


_PACKAGE_ROOT = Path(__file__).resolve().parent
EXAMPLE_RIG_CONFIG = (
    Path("configs/rig.example.yaml")
    if Path("configs/rig.example.yaml").exists()
    else _PACKAGE_ROOT / "configs" / "rig.example.yaml"
)


def load_rig_section(path: Path, section: str) -> dict[str, Any]:
    """Load one mapping from the unified rig YAML."""
    data = load_rig_config(path)
    value = data.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing or invalid '{section}' section in {path}.")
    return value


def load_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> dict[str, Any]:
    """Load the complete rig mapping so optional UX defaults can coexist."""
    if not path.exists():
        raise SystemExit(
            f"Missing rig configuration: {path}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid rig configuration mapping: {path}.")
    return data


def load_optional_rig_section(
    path: Path,
    section: str,
) -> dict[str, Any]:
    """Return an optional rig section without making old rig files invalid."""
    value = load_rig_config(path).get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid '{section}' section in {path}; expected a mapping.")
    return value
