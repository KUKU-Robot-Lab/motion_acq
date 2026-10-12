"""Soft cap on the teleop hand turn (10.11 user, arm tuning).

The hand turns far more than the OpenArm with the RH56F1 can follow: in the
first left trace a 100 deg hand turn left the IK 59 deg / 6 cm off and swung
the shoulder 60 deg. Turns from home now pass 1:1 up to knee x cap and then
saturate toward a per-axis cap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motion_acq.retargeting.handumi_to_robot import limit_target_rotation, soften_rotation
from motion_acq.robots.registry import _rotation_limits, load_embodiment
from motion_acq.teleop.core import TeleopController

POS = np.radians([85.0, 80.0, 50.0])
NEG = np.radians([45.0, 30.0, 45.0])
KNEE = 0.6
TRACKED = {"left": True, "right": True}
OPEN = {"left": 0.0, "right": 0.0}


def turn(axis, deg) -> np.ndarray:
    return np.asarray(axis, dtype=float) * math.radians(deg)


def angle_deg(rotvec) -> float:
    return math.degrees(float(np.linalg.norm(rotvec)))


def test_a_small_turn_passes_unchanged():
    v = turn([0, 0, 1], 25.0)  # below 0.6 x 50
    np.testing.assert_allclose(soften_rotation(v, POS, NEG, KNEE), v, atol=1e-6)


def test_a_big_turn_saturates_below_the_cap_and_keeps_its_axis():
    out = [soften_rotation(turn([0, 0, 1], d), POS, NEG, KNEE) for d in (40, 60, 100, 170)]
    angles = [angle_deg(v) for v in out]
    assert angles == sorted(angles)  # still turns a little more as the hand does
    assert angles[-1] < 50.0 and angles[0] > 30.0
    for v in out:
        np.testing.assert_allclose(v / np.linalg.norm(v), [0, 0, 1], atol=1e-6)


def test_the_cap_depends_on_the_turn_direction():
    up = angle_deg(soften_rotation(turn([0, 1, 0], 90.0), POS, NEG, KNEE))
    down = angle_deg(soften_rotation(turn([0, -1, 0], 90.0), POS, NEG, KNEE))
    assert down < 30.0 < up < 80.0


def test_the_target_keeps_its_position_and_turns_from_home():
    home = np.array([0.1, 0.3, 0.45, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    s = math.sin(math.radians(50.0))
    target = np.array([0.2, 0.1, 0.4, 0.0, 0.0, s, math.cos(math.radians(50.0))], dtype=np.float32)
    out = limit_target_rotation(target, home, POS, NEG, KNEE)  # 100 deg about +z
    np.testing.assert_allclose(out[:3], target[:3])
    yaw = 2.0 * math.degrees(math.atan2(out[5], out[6]))
    assert 30.0 < yaw < 50.0
    np.testing.assert_allclose(out[3:5], 0.0, atol=1e-6)


def test_config_parsing_checks_its_values():
    limits = _rotation_limits({"knee": 0.5, "left": {"pos_deg": [80, 80, 50], "neg_deg": [40, 30, 45]}})
    assert set(limits) == {"left"} and limits["left"].knee == 0.5
    assert _rotation_limits(None) == {}
    with pytest.raises(ValueError):
        _rotation_limits({"left": {"pos_deg": [80, 80], "neg_deg": [40, 30, 45]}})
    with pytest.raises(ValueError):
        _rotation_limits({"knee": 1.2})
    with pytest.raises(ValueError):
        _rotation_limits({"middle": {}})


# The 10.11 left-arm caps, kept here to test the mechanism after 10.12 removed
# them from the robot YAML.
LEFT_CAPS = {"knee": 0.6, "left": {"pos_deg": [85, 80, 50], "neg_deg": [40, 30, 45]}}


def test_the_rh56f1_hand_turns_are_not_capped():
    # 10.12 user: the arms follow the operator's hand as far as they can.
    assert load_embodiment("openarm_rh56f1").config.teleop_rotation_limits == {}
    assert load_embodiment("openarmv1").config.teleop_rotation_limits == {}


def test_the_controller_softens_a_100_deg_hand_turn():
    runtime = load_embodiment("openarm_rh56f1")
    ctl = TeleopController(runtime, home_q=runtime.home_q(), enabled_sides=("left",),
                           source_world_to_robot_world=np.eye(3, dtype=np.float32),
                           translation_scale=0.5)
    ctl.rotation_limits = _rotation_limits(LEFT_CAPS)
    start = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    s = math.sin(math.radians(50.0))
    turned = np.array([0.3, 0.2, 0.1, 0.0, 0.0, s, math.cos(math.radians(50.0))], dtype=np.float32)
    ctl.anchor({"left": start, "right": start}, TRACKED, ("left",))
    target = ctl.step({"left": turned, "right": start}, TRACKED, OPEN).target_pose7["left"]
    home = ctl.anchor_ref["left"]
    rel = _quat_angle_deg(home[3:7], target[3:7])
    assert 30.0 < rel < 50.0  # +z cap for the left arm is 50 deg
    np.testing.assert_allclose(target[:3], home[:3], atol=1e-5)


def _quat_angle_deg(a, b) -> float:
    dot = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
