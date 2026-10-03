"""Lazy registry for optional real-robot teleop adapters."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from motion_acq.real.base import TeleopRobotBackend
from motion_acq.robots.registry import RobotRuntime

RobotBackend = TeleopRobotBackend


_ROBOT_BACKEND_MODULES: dict[str, str] = {
    "openarmv1": "motion_acq.real.openarm.teleop",
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
) -> TeleopRobotBackend:
    """Create a backend without importing SDKs for unused robots."""
    backend_key = _backend_key(robot, runtime)
    try:
        module_name = _ROBOT_BACKEND_MODULES[backend_key]
    except KeyError as exc:
        raise ValueError(
            f"No real hardware backend registered for {robot!r}."
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

__all__ = [
    "REAL_BACKEND_NAMES",
    "RobotBackend",
    "TeleopRobotBackend",
    "make_real_backend",
]
