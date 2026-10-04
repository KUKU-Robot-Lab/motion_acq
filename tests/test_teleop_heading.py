"""Robot forward = where the wearer faces when that arm starts following.

The Quest workspace is set from the first headset pose after the teleop starts
(10.04: 21:22:55, the operator still at the console), 80 s before Space. With
heading_from_hmd the arm takes its forward from the headset heading at its own
anchor, so the moment the teleop process started does not matter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motion_acq.robots.registry import load_embodiment
from motion_acq.teleop.core import TeleopController, heading_world_map
from motion_acq.teleop.session import TeleopInputs, TeleopSession

TRACKED = {"left": True, "right": True}
OPEN = {"left": 0.0, "right": 0.0}


def yaw_pose7(position, yaw_deg: float, pitch_down_deg: float = 0.0) -> np.ndarray:
    """Pose whose +x points at yaw (about +z, left positive), pitched down."""
    y, p = math.radians(yaw_deg) / 2, math.radians(pitch_down_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(y), math.cos(y)])
    qy = np.array([0.0, math.sin(p), 0.0, math.cos(p)])  # +y rotation tips +x down
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]
    return np.array([*position, *q], dtype=np.float32)


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment("openarmv1")


def controller(runtime, heading: bool) -> TeleopController:
    return TeleopController(runtime, home_q=runtime.home_q(), enabled_sides=("left", "right"),
                            source_world_to_robot_world=np.eye(3, dtype=np.float32),
                            translation_scale=1.0, heading_from_hmd=heading)


def robot_step(ctl: TeleopController, hmd: np.ndarray | None, hand_step: np.ndarray) -> np.ndarray:
    start = yaw_pose7([0.4, -0.2, -0.3], 30.0)  # hand-mounted controller, any orientation
    moved = start.copy()
    moved[:3] += hand_step
    ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=hmd)
    target = ctl.step({"left": start, "right": moved}, TRACKED, OPEN).target_pose7["right"]
    return target[:3] - ctl.anchor_ref["right"][:3]


@pytest.mark.parametrize("pitch_down", [0.0, 50.0, 89.0])
def test_moving_where_the_headset_faces_is_robot_forward(runtime, pitch_down):
    ctl = controller(runtime, heading=True)
    hmd = yaw_pose7([0.0, 0.0, 0.0], 90.0, pitch_down)  # turned left, maybe looking down
    np.testing.assert_allclose(robot_step(ctl, hmd, np.array([0.0, 0.05, 0.0])), [0.05, 0.0, 0.0], atol=2e-4)
    np.testing.assert_allclose(robot_step(ctl, hmd, np.array([0.0, 0.0, 0.05])), [0.0, 0.0, 0.05], atol=2e-4)


def test_without_heading_the_workspace_axes_are_used(runtime):
    ctl = controller(runtime, heading=False)
    hmd = yaw_pose7([0.0, 0.0, 0.0], 90.0)
    np.testing.assert_allclose(robot_step(ctl, hmd, np.array([0.0, 0.05, 0.0])), [0.0, 0.05, 0.0], atol=2e-4)


def test_untracked_headset_does_not_start_the_arm(runtime, caplog):
    """Falling back to the workspace axes would be the 10.04 console heading again."""
    ctl = controller(runtime, heading=True)
    start = yaw_pose7([0.4, -0.2, -0.3], 0.0)
    with caplog.at_level("WARNING"):
        assert ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=None) == ()
    assert ctl.anchors["right"] is None
    assert "Headset not tracked" in caplog.text
    nan_hmd = np.full(7, np.nan, dtype=np.float32)
    assert ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=nan_hmd) == ()


def test_the_anchor_logs_the_heading_it_took(runtime, caplog):
    ctl = controller(runtime, heading=True)
    start = yaw_pose7([0.4, -0.2, -0.3], 0.0)
    with caplog.at_level("INFO"):
        ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=yaw_pose7([0, 0, 0], 30.0))
    assert "right" in caplog.text and "+30 deg" in caplog.text


def test_an_arm_already_following_keeps_its_heading(runtime):
    ctl = controller(runtime, heading=True)
    start = yaw_pose7([0.4, 0.2, -0.3], 0.0)
    ctl.anchor({"left": start, "right": start}, TRACKED, ("left",), hmd_pose7=yaw_pose7([0, 0, 0], 0.0))
    ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=yaw_pose7([0, 0, 0], 90.0))
    moved = start.copy()
    moved[0] += 0.05
    target = ctl.step({"left": moved, "right": start}, TRACKED, OPEN).target_pose7["left"]
    np.testing.assert_allclose(target[:3] - ctl.anchor_ref["left"][:3], [0.05, 0.0, 0.0], atol=2e-4)


def test_heading_world_map_is_a_yaw_only_rotation():
    m = heading_world_map(np.eye(3), yaw_pose7([0, 0, 0], 30.0, 40.0))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(m[2], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(m @ np.array([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0]),
                               [1.0, 0.0, 0.0], atol=1e-6)


class Sample:
    left_tcp_pose = right_tcp_pose = yaw_pose7([0.4, -0.2, -0.3], 0.0)
    left_tracked = right_tracked = True
    hmd_pose = yaw_pose7([0.0, 0.0, 0.0], 45.0)
    hmd_tracked = True
    device_time_ns = 1


class Widths:
    left = right = left_normalized = right_normalized = 0.0


def test_session_inputs_carry_the_headset_only_when_tracked():
    inputs: TeleopInputs = TeleopSession.inputs(Sample(), Widths())
    assert inputs.hmd_pose7 is not None
    np.testing.assert_allclose(inputs.hmd_pose7, Sample.hmd_pose)
    lost = Sample()
    lost.hmd_tracked = False
    assert TeleopSession.inputs(lost, Widths()).hmd_pose7 is None


def test_anchor_headings_are_kept_per_arm_for_the_dataset(runtime):
    """Review: the recorded workspace poses need each arm's anchor heading to be re-targeted later."""
    from motion_acq.dataset.raw import anchor_heading_features

    ctl = controller(runtime, heading=True)
    assert np.isnan(ctl.anchor_headings()).all()
    start = yaw_pose7([0.4, -0.2, -0.3], 0.0)
    ctl.anchor({"left": start, "right": start}, TRACKED, ("right",), hmd_pose7=yaw_pose7([0, 0, 0], 30.0))
    left, right = ctl.anchor_headings()
    assert np.isnan(left) and right == pytest.approx(math.radians(30.0), abs=1e-6)
    ctl.park(("right",))
    assert np.isnan(ctl.anchor_headings()).all()
    spec = anchor_heading_features()["observation.tracking.anchor_heading_rad"]
    assert spec["shape"] == (2,) and spec["names"] == ["left", "right"]
