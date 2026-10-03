"""The shared per-frame teleoperation pipeline.

Simulation, physical teleoperation, and recording deliberately share this
module.  Mode-specific code decides *when* an arm may start and what to do
with the resulting command; this module owns the motion mapping itself:
tracking sample -> calibrated hand poses -> anchors -> IK -> smoothed command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from motion_acq.retargeting.handumi_to_robot import raw_state_pose7_pair
from motion_acq.teleop.common import (
    AdaptiveJointFilter,
    sample_state,
    tracking_sample_time_ns,
)
from motion_acq.teleop.core import TeleopController, TeleopStep


@dataclass(frozen=True)
class TeleopInputs:
    """Normalized live inputs, available before an operator-start decision."""

    raw_source_poses: dict[str, np.ndarray]
    side_tracked: dict[str, bool]
    openings: dict[str, float]
    sample_time_ns: int


@dataclass(frozen=True)
class TeleopFrame:
    """One mode-independent result of processing live teleoperation inputs."""

    inputs: TeleopInputs
    source_poses: dict[str, np.ndarray]
    anchored_sides: tuple[str, ...]
    step: TeleopStep
    q: np.ndarray


class TeleopSession:
    """Run the common live-teleoperation motion pipeline once per frame.

    Callers retain ownership of input policy (claps, auto-start, recovery) and
    output policy (rendering, hardware writes, or dataset storage).  Keeping
    those effects outside lets every mode share the safety-critical retargeting
    and smoothing path without making recording depend on a simulator.
    """

    def __init__(
        self,
        controller: TeleopController,
        joint_filter: AdaptiveJointFilter,
    ) -> None:
        self.controller = controller
        self.joint_filter = joint_filter

    @staticmethod
    def inputs(sample: Any, widths: Any) -> TeleopInputs:
        """Normalize a sample once, including data needed by start policies."""
        side_tracked = {
            "left": bool(sample.left_tracked),
            "right": bool(sample.right_tracked),
        }
        state = sample_state(sample, widths)
        raw_source_poses: dict[str, np.ndarray] = dict(
            zip(("left", "right"), raw_state_pose7_pair(state), strict=True)
        )
        return TeleopInputs(
            raw_source_poses=raw_source_poses,
            side_tracked=side_tracked,
            openings={
                "left": float(widths.left_normalized),
                "right": float(widths.right_normalized),
            },
            sample_time_ns=tracking_sample_time_ns(sample),
        )

    def advance(
        self,
        inputs: TeleopInputs,
        *,
        now_s: float,
        start_sides: tuple[str, ...] = (),
        exact_sides: tuple[str, ...] = (),
    ) -> TeleopFrame:
        """Transform normalized inputs into the next robot command."""
        source_poses = {
            side: np.asarray(pose, dtype=np.float32).copy()
            for side, pose in inputs.raw_source_poses.items()
        }
        anchored_sides = self.controller.anchor(
            source_poses, inputs.side_tracked, start_sides
        )
        step = self.controller.step(source_poses, inputs.side_tracked, inputs.openings)
        exact_indices = tuple(
            dict.fromkeys(
                index
                for side in exact_sides
                for index in self.controller.side_indices[side]
            )
        )
        q = self.joint_filter.filter(
            step.q,
            now_s,
            exact_indices=exact_indices,
        )
        return TeleopFrame(
            inputs=inputs,
            source_poses=source_poses,
            anchored_sides=anchored_sides,
            step=step,
            q=q,
        )

    def snapshot_filter(self):
        """Capture the filter before an IK solve that may be superseded."""
        return self.joint_filter.snapshot()

    def restore_filter(self, state) -> None:
        """Discard filter state produced for an unpublished IK solution."""
        self.joint_filter.restore(state)
