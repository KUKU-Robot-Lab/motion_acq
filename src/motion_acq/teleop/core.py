"""Robot-agnostic anchoring and IK state used by simulated and real teleop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import math

import numpy as np

from motion_acq.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
)
from motion_acq.robots.registry import RobotRuntime
from motion_acq.tracking.transforms import heading_yaw

SIDES: tuple[str, str] = ("left", "right")


def heading_world_map(world_map: np.ndarray, hmd_pose7: np.ndarray) -> np.ndarray:
    """``world_map`` turned so the headset heading becomes source forward (+x).

    Only the yaw of the headset counts (z stays up): moving the hand where the
    wearer faces is then robot +x, whatever the workspace heading was.
    """
    yaw = heading_yaw(np.asarray(hmd_pose7, dtype=np.float64)[3:7])
    c, s = math.cos(yaw), math.sin(yaw)
    undo_yaw = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(world_map, dtype=np.float64) @ undo_yaw).astype(np.float32)


def _usable_pose7(pose7: np.ndarray | None) -> bool:
    if pose7 is None:
        return False
    pose = np.asarray(pose7, dtype=np.float64)
    return pose.shape == (7,) and bool(np.all(np.isfinite(pose))) and np.linalg.norm(pose[3:7]) > 1e-6


@dataclass(frozen=True)
class TeleopStep:
    """One solved teleoperation command plus state-transition information."""

    q: np.ndarray
    anchored_sides: tuple[str, ...]
    target_pose7: dict[str, np.ndarray]


class TeleopController:
    """Own anchors, home policy, and the shared bimanual IK state.

    Tracking, rendering, and hardware I/O deliberately stay outside this class.
    Both frontends therefore apply the exact same motion mapping while retaining
    their environment-specific lifecycle and diagnostics.
    """

    def __init__(
        self,
        runtime: RobotRuntime,
        *,
        home_q: np.ndarray,
        enabled_sides: tuple[str, ...],
        source_world_to_robot_world: np.ndarray,
        translation_scale: float,
        anchor_z: float | None = None,
        heading_from_hmd: bool = False,
    ) -> None:
        self.runtime = runtime
        self.solver = runtime.solver_cls()
        self.home_q = np.asarray(home_q, dtype=np.float32).copy()
        self.q = self.home_q.copy()
        self.enabled_sides = enabled_sides
        self.source_world_to_robot_world = np.asarray(
            source_world_to_robot_world, dtype=np.float32
        )
        self.translation_scale = float(translation_scale)
        # True: each arm takes robot forward from the headset heading at its anchor
        self.heading_from_hmd = bool(heading_from_hmd)
        self.max_reach = runtime.config.ik_weights.max_reach
        self.side_indices = {side: runtime.arm_joint_indices(side) for side in SIDES}
        left_home, right_home = self.solver.fk_pose7(self.home_q)
        self.home_pose7 = {"left": left_home, "right": right_home}
        self.anchor_ref = {side: pose.copy() for side, pose in self.home_pose7.items()}
        if anchor_z is not None:
            for pose in self.anchor_ref.values():
                pose[2] = float(anchor_z)
        self.anchors: dict[str, dict[str, np.ndarray] | None] = {
            side: None for side in SIDES
        }
        self.tracking_hold_sides: set[str] = set()

    def warmup(self) -> None:
        """Compile/warm the solver without touching a renderer or robot."""
        self.solver.ik(
            self.q,
            left_pose=self._pose_target(self.home_pose7["left"]),
            right_pose=self._pose_target(self.home_pose7["right"]),
        )

    @staticmethod
    def _pose_target(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pose7[:3], pose7[3:7]

    @property
    def active(self) -> bool:
        return any(self.anchors[side] is not None for side in self.enabled_sides)

    def idle_sides(self) -> tuple[str, ...]:
        return tuple(side for side in self.enabled_sides if self.anchors[side] is None)

    def reset(self) -> np.ndarray:
        for side in self.enabled_sides:
            self.anchors[side] = None
        self.tracking_hold_sides.difference_update(self.enabled_sides)
        self.q = self.home_q.copy()
        return self.q.copy()

    def park(self, sides: tuple[str, ...]) -> tuple[str, ...]:
        """Deactivate selected arms and send their joints back to home.

        Unlike :meth:`reset`, this preserves the anchors and joint commands of
        every other arm so a bimanual teleop session can park one side without
        interrupting the other.
        """
        parked: list[str] = []
        for side in sides:
            if side not in self.enabled_sides:
                continue
            self.anchors[side] = None
            self.tracking_hold_sides.discard(side)
            indices = self.side_indices[side]
            self.q[indices] = self.home_q[indices]
            parked.append(side)
        return tuple(parked)

    def tracking_lost(self, held_q: np.ndarray) -> None:
        """Cancel motion and keep recovered arms at the backend's held pose."""
        for side in self.enabled_sides:
            self.anchors[side] = None
        self.tracking_hold_sides.update(self.enabled_sides)
        self.q = np.asarray(held_q, dtype=np.float32).copy()

    def anchor(
        self,
        source_poses: Mapping[str, np.ndarray],
        side_tracked: Mapping[str, bool],
        requested_sides: tuple[str, ...],
        *,
        hmd_pose7: np.ndarray | None = None,
    ) -> tuple[str, ...]:
        """Start the requested arms from their current hand poses.

        With heading_from_hmd and a tracked headset, robot forward is where the
        headset faces now; else the workspace axes. Arms already following keep
        the mapping of their own anchor.
        """
        world_map = self.source_world_to_robot_world
        if self.heading_from_hmd and _usable_pose7(hmd_pose7):
            world_map = heading_world_map(world_map, np.asarray(hmd_pose7))
        anchored: list[str] = []
        for side in requested_sides:
            if side not in self.enabled_sides or not side_tracked[side]:
                continue
            source = np.asarray(source_poses[side], dtype=np.float32)
            self.anchors[side] = {
                "source": source.copy(),
                "adapter": local_frame_adapter(
                    source,
                    self.anchor_ref[side],
                    source_world_to_robot_world=world_map,
                ),
            }
            self.tracking_hold_sides.discard(side)
            anchored.append(side)
        return tuple(anchored)

    def step(
        self,
        source_poses: Mapping[str, np.ndarray],
        side_tracked: Mapping[str, bool],
        gripper_openings: Mapping[str, float],
    ) -> TeleopStep:
        targets: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
            side: None for side in SIDES
        }
        target_pose7: dict[str, np.ndarray] = {}
        for side in SIDES:
            anchor = self.anchors[side]
            if anchor is None or not side_tracked[side]:
                continue
            pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=anchor["source"],
                current_source_pose7=source_poses[side],
                base_robot_pose7=self.anchor_ref[side],
                adapter_rot=anchor["adapter"],
                home_robot_pose7=self.anchor_ref[side],
                translation_scale=self.translation_scale,
                max_reach=self.max_reach,
            )
            targets[side] = self._pose_target(pose7)
            target_pose7[side] = pose7

        previous_q = self.q.copy()
        self.q = self.solver.ik(
            self.q,
            left_pose=targets["left"],
            right_pose=targets["right"],
        )
        for side in SIDES:
            if self.anchors[side] is not None:
                continue
            source = previous_q if side in self.tracking_hold_sides else self.home_q
            self.q[self.side_indices[side]] = source[self.side_indices[side]]
        self.runtime.set_finger_positions(self.q, gripper_openings)
        return TeleopStep(
            q=self.q.copy(),
            anchored_sides=tuple(
                side for side in SIDES if self.anchors[side] is not None
            ),
            target_pose7=target_pose7,
        )
