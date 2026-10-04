"""Simulated OpenArm backend: the real OpenArm driver on top of a fake SDK (no CAN).

Selected with --fake-robot on teleop-real / teleop-record. Everything above
the SDK is the production path: settings from the station rig (J8 on/off,
ports, gains, speed limits, watchdog, following error), startup read, slow
home, streamer. Only the CAN link check is skipped.
MACQ_FAKE_ROBOT_START=home starts the motors at the station home pose
instead of zeros (shorter fake runs).
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

from motion_acq.real.base import TeleopRobotBackend
from motion_acq.real.openarm.driver import (
    OpenArmCanEnvironment,
    OpenArmSdkSide,
    load_gravity_models,
    load_openarm_settings,
)
from motion_acq.real.openarm.fake_sdk import ARM_DOF, FakeOpenArmSdk
from motion_acq.real.openarm.gripper_calibration import (
    user_openarm_gripper_calibration_path,
)
from motion_acq.real.openarm.teleop import OpenArmBackend
from motion_acq.robots.registry import RobotRuntime, resolve_home_q

log = logging.getLogger(__name__)


class FakeOpenArmBackend(OpenArmBackend):
    name = "openarmv1-fake"

    def setup(self, *, repair: bool = True) -> None:
        log.warning("FAKE ROBOT: simulated OpenArm SDK, no CAN interface is opened.")


def _start_q_by_port(runtime: RobotRuntime, rig_config: Path, settings) -> dict[str, list[float]]:
    if os.environ.get("MACQ_FAKE_ROBOT_START", "zero") != "home":
        return {}
    _, home = resolve_home_q(runtime, rig_config=rig_config)
    names = list(runtime.joint_names)
    ports = {"left": settings.left_port, "right": settings.right_port}
    return {
        ports[side]: [float(home[names.index(f"openarm_{side}_joint{i}")]) for i in range(1, ARM_DOF + 1)]
        for side in ("left", "right")
    }


def build_backend(
    *,
    runtime: RobotRuntime,
    rig_config: Path,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> TeleopRobotBackend:
    settings = load_openarm_settings(
        rig_config, runtime.config.real_options, user_openarm_gripper_calibration_path(),
        robot_name=runtime.name,
    )
    # The fake arm sags under the same gravity model the real one has (when the
    # robot YAML gives one), so the feedforward is exercised, not assumed.
    gravity = load_gravity_models(settings, ("left", "right"))
    ports = {"left": settings.left_port, "right": settings.right_port}
    sdk = FakeOpenArmSdk(
        start_q_by_port=_start_q_by_port(runtime, rig_config, settings),
        gravity_by_port={ports[side]: model for side, model in gravity.items()},
    )
    environment = OpenArmCanEnvironment(
        settings,
        active_sides=active_sides,
        joint_limits={
            name: (float(lower), float(upper))
            for name, lower, upper in zip(
                runtime.joint_names,
                runtime.robot.joints.lower_limits,
                runtime.robot.joints.upper_limits,
                strict=True,
            )
        },
        side_factory=functools.partial(OpenArmSdkSide, sdk=sdk),
    )
    backend = FakeOpenArmBackend(environment, joint_names=runtime.joint_names, active_sides=active_sides)
    backend.fake_sdk = sdk
    return backend


__all__ = ["FakeOpenArmBackend", "build_backend"]
