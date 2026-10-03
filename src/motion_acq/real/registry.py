"""Lazy registry for optional real-robot teleop adapters."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml

from motion_acq.real.base import TeleopRobotBackend
from motion_acq.robots.registry import RobotRuntime

RobotBackend = TeleopRobotBackend


_ROBOT_BACKEND_MODULES: dict[str, str] = {
    "openarmv1": "motion_acq.real.openarm.teleop",
}

# --fake-robot: same backend key, simulated SDK (no CAN).
_FAKE_BACKEND_MODULES: dict[str, str] = {
    "openarmv1": "motion_acq.real.openarm.fake",
}

_REAL_BACKEND_ALIASES: dict[str, str] = {
    "openarm_can": "openarmv1",
}


def make_real_backend(
    robot: str,
    *,
    runtime: RobotRuntime,
    rig_config: Path,
    active_sides: tuple[str, ...] = ("left", "right"),
    fake: bool = False,
) -> TeleopRobotBackend:
    """Create a backend without importing SDKs for unused robots."""
    backend_key = _backend_key(robot, runtime)
    modules = _FAKE_BACKEND_MODULES if fake else _ROBOT_BACKEND_MODULES
    try:
        module_name = modules[backend_key]
    except KeyError as exc:
        raise ValueError(
            f"No {'fake' if fake else 'real hardware'} backend registered for {robot!r}."
        ) from exc
    module = import_module(module_name)
    return module.build_backend(
        runtime=runtime,
        rig_config=rig_config,
        active_sides=active_sides,
    )


def _backend_key(robot: str, runtime: RobotRuntime) -> str:
    configured = runtime.config.real_options.get("backend")
    key = str(configured or robot)
    return _REAL_BACKEND_ALIASES.get(key, key)


REAL_BACKEND_NAMES: tuple[str, ...] = tuple(sorted(_ROBOT_BACKEND_MODULES))


def _robots_with_backend() -> tuple[str, ...]:
    """Robot YAMLs whose real.backend resolves to a registered backend."""
    from motion_acq.robots.registry import CONFIG_DIR, available_robot_names

    names = []
    for name in available_robot_names():
        data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}
        backend = str(((data.get("real") or {}).get("backend")) or name)
        if _REAL_BACKEND_ALIASES.get(backend, backend) in _ROBOT_BACKEND_MODULES:
            names.append(name)
    return tuple(names)


REAL_ROBOT_NAMES: tuple[str, ...] = _robots_with_backend()

__all__ = [
    "REAL_BACKEND_NAMES",
    "REAL_ROBOT_NAMES",
    "RobotBackend",
    "TeleopRobotBackend",
    "make_real_backend",
]
