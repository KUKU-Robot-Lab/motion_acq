"""The workspace reset must keep the robot frame level, whatever the head is doing.

The arm teleop re-centres the Quest workspace on the headset pose the first time it
sees one. If that pose is used whole, a wearer who happens to look down at that moment
tilts the whole workspace: reaching forward then drives the arm up. Only the heading
(yaw about gravity) of the headset may define the workspace axes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motion_acq.head.retarget import hmd_yaw_pitch_deg
from motion_acq.tracking.meta_quest import HmdState, workspace_from_hmd
from motion_acq.tracking.transforms import quat_multiply, unity_pose_to_handumi

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def unity_quat(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Unity ``Quaternion.Euler(pitch, yaw, roll)``: +yaw turns right, +pitch looks down."""

    def axis(index: int, deg: float) -> np.ndarray:
        q = np.zeros(4)
        q[index] = math.sin(math.radians(deg) / 2.0)
        q[3] = math.cos(math.radians(deg) / 2.0)
        return q

    return quat_multiply(axis(1, yaw_deg), quat_multiply(axis(0, pitch_deg), axis(2, roll_deg)))


def workspace_step(workspace, start_unity: np.ndarray, step_unity: np.ndarray) -> np.ndarray:
    """Workspace displacement of a controller moved by ``step_unity`` (raw Unity metres)."""
    before = workspace.apply(unity_pose_to_handumi(start_unity, IDENTITY_Q))
    after = workspace.apply(unity_pose_to_handumi(start_unity + step_unity, IDENTITY_Q))
    return after.position - before.position


@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0])
@pytest.mark.parametrize(("pitch_deg", "roll_deg"), [(0.0, 0.0), (40.0, 0.0), (-25.0, 0.0), (55.0, 12.0)])
def test_reaching_forward_stays_forward_when_the_head_is_tilted(yaw_deg, pitch_deg, roll_deg):
    hmd = HmdState(
        tracked=True,
        position=np.array([0.1, 1.6, -0.2]),
        quaternion=unity_quat(yaw_deg, pitch_deg, roll_deg),
    )
    workspace = workspace_from_hmd(hmd)

    yaw = math.radians(yaw_deg)
    facing = np.array([math.sin(yaw), 0.0, math.cos(yaw)])  # Unity: where the wearer faces
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    start = np.array([0.3, 1.1, 0.4])

    np.testing.assert_allclose(workspace_step(workspace, start, 0.1 * facing), [0.1, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(workspace_step(workspace, start, 0.1 * right), [0.0, -0.1, 0.0], atol=1e-9)
    np.testing.assert_allclose(workspace_step(workspace, start, 0.1 * up), [0.0, 0.0, 0.1], atol=1e-9)


def test_the_headset_itself_keeps_its_pitch_in_the_workspace():
    hmd = HmdState(tracked=True, position=np.array([0.0, 1.6, 0.0]), quaternion=unity_quat(30.0, 40.0))
    workspace = workspace_from_hmd(hmd)
    head = workspace.apply(unity_pose_to_handumi(hmd.position, hmd.quaternion))

    np.testing.assert_allclose(head.position, [0.0, 0.0, 0.0], atol=1e-9)
    yaw, pitch = hmd_yaw_pitch_deg(np.r_[head.position, head.quaternion])
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(-40.0, abs=1e-6)  # still looking down, only the heading is removed


def test_looking_straight_down_still_gives_a_level_workspace():
    hmd = HmdState(tracked=True, position=np.array([0.0, 1.6, 0.0]), quaternion=unity_quat(20.0, 90.0))
    workspace = workspace_from_hmd(hmd)
    up = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(workspace_step(workspace, np.zeros(3), 0.1 * up), [0.0, 0.0, 0.1], atol=1e-9)
