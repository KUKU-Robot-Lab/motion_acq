"""Joystick map (10.11 user): each robot joint from one glove angle of that finger, tuned in a window."""

from __future__ import annotations

import pytest
from hand_fixtures import CONFIG, make_calibration

from motion_acq.hand.calibration import CalibrationError, HandCalibration
from motion_acq.hand.joystick import (
    JOINTS, POINTS, JointMap, JoystickError, JoystickMap, JoystickRetargeter, default_joint)
from motion_acq.hand.retarget import HandRetargeter
from motion_acq.hand.rh56f1 import load_rh56f1_map
from motion_acq.hand.synthetic import synthetic_angles

HAND_MAP = load_rh56f1_map()
TARGETS = {p: e.target for p, e in CONFIG.examples.items()}
CLOSED = CONFIG.limits_rad["index_1"][1]


def finger(source: str) -> JointMap:
    return JointMap(source, (0.12, 3.14), (0.0, CLOSED))


# thumb_1 from thumb_mcp: spread -0.2 -> 0.8 rad, beside the index 0.0 -> 1.57, across the palm 0.4 -> 2.0
THUMB_1 = JointMap("thumb_mcp", (-0.2, 0.0, 0.4), (0.8, 1.57, 2.0))
THUMB_2 = JointMap("thumb_pip", (0.01, 0.72), (0.0, CONFIG.limits_rad["thumb_2"][1]))
MAP = JoystickMap({"thumb_1": THUMB_1, "thumb_2": THUMB_2, "index_1": finger("index_mcp+index_pip"),
                   "middle_1": finger("middle_mcp+middle_pip"), "ring_1": finger("ring_mcp+ring_pip")})


def test_piecewise_linear_through_the_points_and_clipped():
    assert THUMB_1.angle(-0.2) == pytest.approx(0.8)
    assert THUMB_1.angle(-0.1) == pytest.approx((0.8 + 1.57) / 2)
    assert THUMB_1.angle(0.0) == pytest.approx(1.57)
    assert THUMB_1.angle(0.2) == pytest.approx((1.57 + 2.0) / 2)
    assert THUMB_1.angle(-5.0) == pytest.approx(0.8) and THUMB_1.angle(5.0) == pytest.approx(2.0)
    falling = JointMap("thumb_brake", (0.9, 0.3), (0.0, 1.0))  # a reading that falls as the joint closes
    assert falling.angle(0.9) == pytest.approx(0.0) and falling.angle(0.6) == pytest.approx(0.5)
    assert falling.angle(0.0) == pytest.approx(1.0)
    assert THUMB_1.glove_at(1.57) == pytest.approx(0.0)
    assert THUMB_1.glove_at(3.0) is None


def test_points_must_run_one_way():
    """The left glove's thumb_brake read open 0.53, flat 0.31, opposed 0.81: no joystick through those."""
    with pytest.raises(JoystickError, match="한 방향"):
        JoystickMap({"thumb_1": JointMap("thumb_brake", (0.53, 0.31, 0.81), (0.8, 1.57, 2.0))})
    with pytest.raises(JoystickError, match="한 방향"):  # two points on the same reading
        JoystickMap({"index_1": JointMap("index_pip", (0.5, 0.52), (0.0, 1.0))})
    with pytest.raises(JoystickError, match="points expected"):
        JoystickMap({"thumb_1": JointMap("thumb_mcp", (0.0, 0.4), (1.57, 2.0))})
    with pytest.raises(JoystickError, match="unknown"):
        JoystickMap({"wrist": finger("index_pip")})


def test_one_glove_finger_moves_one_robot_joint():
    """10.11 user: the thumb only from the Nova's thumb, the index only from the index."""
    joy = JoystickRetargeter(MAP, CONFIG.home_rad)
    base = joy.solve(synthetic_angles(0.0, 0.0, 0.0))
    assert base["thumb_1"] == pytest.approx(1.57, abs=1e-6)
    assert all(base[j] == pytest.approx(0.0, abs=0.01) for j in ("thumb_2", "index_1", "middle_1", "ring_1"))

    def moved(signals) -> set[str]:
        q = joy.solve(signals)
        return {j for j in q if abs(q[j] - base[j]) > 1e-6}

    assert moved(synthetic_angles(0.0, 0.0, 1.0)) == {"thumb_1"}  # swing across the palm
    assert moved(synthetic_angles(0.0, 1.0, 0.0)) == {"thumb_2"}  # tip bend only
    assert moved(synthetic_angles({"index": 1.0}, 0.0, 0.0)) == {"index_1"}
    assert moved(synthetic_angles({"ring": 1.0}, 0.0, 0.0)) == {"ring_1", "pinky_1"}  # pinky follows the ring
    assert joy.solve(synthetic_angles(0.0, 0.0, 1.0))["thumb_1"] == pytest.approx(2.0)


def test_reference_take_shifts_a_shifted_glove_only():
    joy = JoystickRetargeter(MAP, CONFIG.home_rad)
    shifted = dict(synthetic_angles(0.0, 0.0, 0.0), thumb_mcp=0.1, index_mcp=0.05 + 0.3)
    assert sorted(joy.take_reference([shifted] * 10)) == sorted(JOINTS)
    q = joy.solve(shifted)
    assert q["thumb_1"] == pytest.approx(1.57) and q["index_1"] == pytest.approx(0.0, abs=1e-6)
    assert joy.maps["thumb_1"].glove == pytest.approx((-0.1, 0.1, 0.5))
    joy.reset()
    bent = synthetic_angles({"index": 1.0}, 0.0, 0.0)  # a bent finger is not a shifted glove
    assert "index_1" not in joy.take_reference([bent] * 10)
    assert joy.maps["index_1"] == MAP.joints["index_1"]


def test_defaults_come_from_the_calibration_poses():
    cal = make_calibration("right")
    m = default_joint("index_1", cal.medians, TARGETS)
    assert m is not None and m.source == "index_mcp+index_pip"
    assert m.robot == (0.0, pytest.approx(CLOSED))
    assert m.glove[0] < 0.5 < 2.5 < m.glove[1]
    assert default_joint("thumb_1", cal.medians, TARGETS).robot == (0.8, 1.57, 2.0)
    assert default_joint("thumb_1", {"open": {}}, TARGETS) is None
    assert [p[1] for p in POINTS["thumb_1"]] == ["open", "flat", "thumb_opposed"]


def test_saved_in_the_calibration(tmp_path):
    cal = make_calibration("right").with_joystick(MAP)
    path = tmp_path / "op_right.yaml"
    cal.save(path)
    loaded = HandCalibration.load(path, side="right")
    assert loaded.joystick == MAP
    raw = path.read_text().replace("- -0.2", "- 0.3")  # thumb_1's points out of order
    path.write_text(raw)
    with pytest.raises(CalibrationError, match="한 방향"):
        HandCalibration.load(path, side="right")


def test_hand_retargeter_uses_the_joystick_map():
    cal = make_calibration("right")
    plain = HandRetargeter(CONFIG, cal, HAND_MAP, "right")
    joy = HandRetargeter(CONFIG, cal.with_joystick(MAP), HAND_MAP, "right")
    for r in (plain, joy):
        r.start(None, 0.0)
    signals = synthetic_angles(0.0, 1.0, 0.0)  # the thumb tip bent, the thumb beside the index
    raw, _ = joy.target(signals)
    assert joy.method_used == "joystick"
    assert raw["thumb_1"] == pytest.approx(1.57) and raw["thumb_2"] == pytest.approx(CONFIG.limits_rad["thumb_2"][1])
    step = joy.step(signals, 0.01)
    assert "joy_thumb_1" in step.features and step.features["joy_lo_thumb_1"] == pytest.approx(-0.2)
    # a partial map: only the thumb from the joystick, the fingers from the configured method
    part = HandRetargeter(CONFIG, cal.with_joystick(JoystickMap({"thumb_1": THUMB_1})), HAND_MAP, "right")
    part.start(None, 0.0)
    raw, _ = part.target(synthetic_angles({"index": 1.0}, 0.0, 1.0))
    assert part.method_used == "flexion+joystick"
    assert raw["thumb_1"] == pytest.approx(2.0) and raw["index_1"] > 1.0
