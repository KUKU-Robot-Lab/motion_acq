"""HandUMI teleop backend adapter for OpenArm v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_acq.real.base import TeleopRobotBackend
from motion_acq.real.openarm.driver import OpenArmCanEnvironment, load_openarm_settings
from motion_acq.real.openarm.gripper_calibration import (
    user_openarm_gripper_calibration_path,
)
from motion_acq.robots.registry import RobotRuntime


class OpenArmBackend:
    """Robot-neutral teleop wrapper around ``OpenArmCanEnvironment``."""

    name = "openarmv1"

    def __init__(
        self,
        environment: OpenArmCanEnvironment,
        *,
        joint_names: tuple[str, ...],
        active_sides: tuple[str, ...] = ("left", "right"),
    ) -> None:
        self.environment = environment
        self.joint_names = tuple(joint_names)
        self.active_sides = tuple(active_sides)
        self._last_q: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        *,
        runtime: RobotRuntime,
        rig_config: Path,
        active_sides: tuple[str, ...] = ("left", "right"),
    ) -> OpenArmBackend:
        environment = OpenArmCanEnvironment(
            load_openarm_settings(
                rig_config,
                runtime.config.real_options,
                user_openarm_gripper_calibration_path(),
                robot_name=runtime.name,
            ),
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
        )
        return cls(
            environment,
            joint_names=runtime.joint_names,
            active_sides=active_sides,
        )

    def setup(self, *, repair: bool = True) -> None:
        self.environment.prepare(repair=repair)

    def connect(self) -> None:
        self.environment.connect()

    def disconnect(self) -> None:
        self.environment.close()

    def read(self, base_q: np.ndarray | None = None) -> np.ndarray:
        base = self._base_q(base_q)
        if self.environment.streamer is None:
            return base
        return self.environment._merge_q(
            self.environment.streamer.feedback(),
            base,
            list(self.joint_names),
        )

    def read_gripper_openings(self) -> dict[str, float]:
        # The current OpenArm SDK wrapper does not expose measured gripper
        # position. An empty result disables close-to-park instead of treating
        # a commanded opening as physical feedback.
        return {}

    def hand_phase(self, phase: str, sides: tuple[str, ...]) -> None:
        """RH56F1 hands (motion_acq.hand_link): teleop = the arm follows the operator, home = parked."""
        self.environment.hand_phase(phase, tuple(sides))

    def home(self, q: np.ndarray) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.home(q, list(self.joint_names))

    def move_home(self, q: np.ndarray) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.move_home(q, list(self.joint_names))

    def rest(self, q: np.ndarray) -> None:
        """From home along the stored path to rest (all 0) before the motors go off."""
        self.environment.rest(q, list(self.joint_names))

    def write(
        self,
        q: np.ndarray,
        gripper_openings: dict[str, float],
    ) -> None:
        self._last_q = np.asarray(q, dtype=np.float32).copy()
        self.environment.command(q, list(self.joint_names), gripper_openings)

    def hold(self, base_q: np.ndarray) -> np.ndarray:
        q = self.environment.hold(base_q, list(self.joint_names))
        self._last_q = q.copy()
        return q

    def check_health(self) -> None:
        self.environment.check_health()

    def _base_q(self, base_q: np.ndarray | None) -> np.ndarray:
        if base_q is not None:
            return np.asarray(base_q, dtype=np.float32).copy()
        if self._last_q is not None:
            return self._last_q.copy()
        return np.zeros(len(self.joint_names), dtype=np.float32)


def build_backend(
    *,
    runtime: RobotRuntime,
    rig_config: Path,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> TeleopRobotBackend:
    return OpenArmBackend.from_config(
        runtime=runtime,
        rig_config=rig_config,
        active_sides=active_sides,
    )


__all__ = ["OpenArmBackend", "build_backend"]
