"""Flexion retargeting (10.11 user: every finger like a one-joint gripper showing the bend)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hand_fixtures import CONFIG, make_calibration

from motion_acq.hand.calibration import rezero
from motion_acq.hand.flexion import FLEX_INPUTS, FlexionConfig, FlexionRetargeter, flexion_config, reading
from motion_acq.hand.kinematic import glove_points
from motion_acq.hand.retarget import HandRetargeter
from motion_acq.hand.rh56f1 import load_rh56f1_map
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles, synthetic_hand_model

HAND_MAP = load_rh56f1_map()
CLOSED = {j: CONFIG.limits_rad[j][1] for j in FLEX_INPUTS}
FINGERS = ("index_1", "middle_1", "ring_1", "pinky_1")


@pytest.fixture(scope="module")
def cal():
    return make_calibration("left")


def flexion(cal, **changes) -> FlexionRetargeter:
    return FlexionRetargeter(cal.medians, CLOSED, dataclasses.replace(CONFIG.flexion, **changes))


def hold(fx: FlexionRetargeter, signals, n: int = 200) -> dict[str, float]:
    out = {}
    for _ in range(n):
        out = fx.solve(signals)
    return out


def test_configured_method_is_flexion():
    assert CONFIG.method == "flexion"
    with pytest.raises(ValueError, match="unknown keys"):
        flexion_config({"finger_span": 1.0})
    with pytest.raises(ValueError):
        FlexionConfig(finger_min_span=0.1, finger_deadband=0.2)


def test_open_hand_is_straight_and_fist_is_closed(cal):
    fx = flexion(cal)
    assert hold(fx, POSE_ANGLES["flat"]) == pytest.approx({j: 0.0 for j in FLEX_INPUTS}, abs=1e-6)
    q = hold(fx, POSE_ANGLES["fist"])  # (the calibration's fist is a median of noisy samples)
    assert q == pytest.approx(CLOSED, abs=0.01)


@pytest.mark.parametrize("finger,joint", [("index", "index_1"), ("middle", "middle_1"), ("ring", "ring_1")])
def test_each_finger_shows_its_own_bend(cal, finger, joint):
    """The middle finger closes all the way too (10.11: the fingertip match stopped it at 0.28 rad)."""
    fx = flexion(cal)
    prev = -1.0
    for curl in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = hold(fx, synthetic_angles({finger: curl}, 0.0, 0.0), n=5)
        assert q[joint] >= prev - 1e-9
        prev = q[joint]
        others = [j for j in FINGERS if j != joint and not (joint == "ring_1" and j == "pinky_1")]
        assert all(q[j] == 0.0 for j in others)
        assert q["thumb_2"] == 0.0
    assert prev == pytest.approx(CLOSED[joint], abs=0.01)
    half = hold(fx, synthetic_angles({finger: 0.5}, 0.0, 0.0), n=5)[joint]
    assert 0.3 * CLOSED[joint] < half < 0.6 * CLOSED[joint]


def test_pinky_follows_the_ring_finger(cal):
    """Nova 2 has no little-finger sensor."""
    q = hold(flexion(cal), synthetic_angles({"ring": 0.7}, 0.0, 0.0), n=5)
    assert q["pinky_1"] == q["ring_1"] > 0.5


def test_thumb_2_follows_the_thumb_bend_alone(cal):
    """10.11: thumb_2 barely moved and only with thumb_1; now the thumb's own bend drives it."""
    fx = flexion(cal)
    bent = hold(fx, synthetic_angles(0.0, 1.0, 0.0), n=5)["thumb_2"]
    assert bent == pytest.approx(CLOSED["thumb_2"], abs=0.01)
    swung = hold(fx, synthetic_angles(0.0, 0.0, 1.0), n=5)["thumb_2"]  # rotation without a bend
    assert swung == 0.0
    assert 0.0 < hold(fx, synthetic_angles(0.0, 0.5, 0.0), n=5)["thumb_2"] < bent


def test_open_end_follows_a_held_open_hand_but_not_a_glitch(cal):
    fx = flexion(cal, finger_max_span=4.0)  # (the synthetic fist reads 3.14: room below the flat hand)
    j = "middle_1"
    start = fx.open[j]
    lower = {**POSE_ANGLES["flat"], "middle_mcp": POSE_ANGLES["flat"]["middle_mcp"] - 0.4}
    fx.solve(lower)  # one sample: the smoothed reading barely moves
    fx.solve(POSE_ANGLES["flat"])
    assert fx.open[j] > start - 0.05
    hold(fx, lower, n=400)  # held: the open end comes down to it
    assert fx.open[j] == pytest.approx(reading(j, lower), abs=0.01) and fx.open[j] < start - 0.1
    assert hold(fx, lower, n=5)[j] == 0.0
    assert hold(fx, POSE_ANGLES["flat"], n=5)[j] > 0.0  # the old open hand now reads a little bent
    hold(fx, POSE_ANGLES["fist"])
    assert fx.open[j] == pytest.approx(reading(j, lower), abs=0.01)  # a fist never moves it back up
    way_down = {**POSE_ANGLES["flat"], "middle_mcp": -10.0}
    hold(fx, way_down, n=400)
    assert fx.open[j] == pytest.approx(fx.closed[j] - 4.0)
    fx.reset()
    assert fx.open[j] == pytest.approx(start)


def test_reference_take_sets_a_shifted_open_hand_but_not_a_bent_one(cal):
    shifted = {k: v + (0.2 if k.endswith(("_mcp", "_pip")) else 0.0) for k, v in POSE_ANGLES["flat"].items()}
    fx = flexion(cal)
    # fingers +0.4 (mcp+pip) is a shift; the thumb +0.2 is past its 0.15 tolerance: a bent thumb
    assert sorted(fx.take_reference([shifted] * 30)) == sorted(FINGERS)
    assert fx.open["index_1"] == pytest.approx(reading("index_1", shifted))
    q = hold(fx, shifted, n=5)
    assert q == pytest.approx({**{j: 0.0 for j in FINGERS}, "thumb_2": q["thumb_2"]}, abs=1e-6)
    assert q["thumb_2"] > 0.0
    bent = synthetic_angles(0.5, 0.0, 0.0)  # a half-curled hand at the reference: not a shift
    fx = flexion(cal)
    open_before = dict(fx.open)
    assert "index_1" not in fx.take_reference([bent] * 30)
    assert fx.open["index_1"] == open_before["index_1"]
    assert hold(fx, bent, n=5)["index_1"] > 0.5


def with_model(angles, side="left"):
    hand, tips = synthetic_hand_model(angles, side)
    return {**angles, **glove_points(hand, tips)}


def test_retargeter_flexion_fingers_and_thumb_rotation_from_the_tips():
    cal = rezero(make_calibration("left"), [with_model(POSE_ANGLES["open"])] * 10)
    rt = HandRetargeter(CONFIG, cal, HAND_MAP, "left")
    rt.start(None, 0.0)
    _, q = rt.target(with_model(synthetic_angles({"middle": 1.0}, 0.0, 0.0)))
    assert rt.method_used == "flexion"
    assert q["middle_1"] == pytest.approx(CLOSED["middle_1"], abs=0.01) and q["index_1"] == 0.0
    assert rt.flexion is not None
    opposed = synthetic_angles(0.0, 0.3, 1.0)
    _, q = rt.target(with_model(opposed))
    assert q["thumb_1"] > 1.85  # the thumb across the palm: rotation from the fingertip match
    assert q["thumb_2"] == pytest.approx(rt.flexion.closure("thumb_2", opposed["thumb_pip"]) * CLOSED["thumb_2"])
    step = None
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        step = rt.step(with_model(POSE_ANGLES["flat"]), t)
    assert step is not None and step.features["flex_open_middle_1"] == pytest.approx(rt.flexion.open["middle_1"])


def test_without_finger_angles_the_example_map_follows(cal):
    rt = HandRetargeter(CONFIG, cal, HAND_MAP, "left")
    rt.start(None, 0.0)
    signals = {k: v for k, v in POSE_ANGLES["fist"].items() if k != "middle_pip"}
    rt.target(signals)
    assert rt.method_used == "examples"


def test_thumb_2_fixed_keeps_the_solved_thumb_on_that_row():
    from motion_acq.hand.kinematic import KinematicRetargeter, TipTables, points_from_signals, table_path

    cal = rezero(make_calibration("left"), [with_model(POSE_ANGLES["open"])] * 10)
    assert cal.alignment is not None
    kin = KinematicRetargeter(TipTables.load(table_path("left")), cal.alignment, CONFIG.kinematic)
    pts = points_from_signals(with_model(synthetic_angles(0.0, 1.0, 0.0)))
    assert pts is not None
    q = kin.solve(pts, fixed={"thumb_2": 0.1, "index_1": 0.5})
    assert q["thumb_2"] == pytest.approx(0.1, abs=0.003) and q["index_1"] == pytest.approx(0.5, abs=0.01)
    assert np.isfinite(q["thumb_1"])


def test_a_finger_moving_toward_the_thumb_does_not_turn_the_robot_thumb():
    """10.11 user (joystick): the thumb from the operator's thumb alone; the pinch term turned it."""
    cal = rezero(make_calibration("left"), [with_model(POSE_ANGLES["open"])] * 10)
    rt = HandRetargeter(CONFIG, cal, HAND_MAP, "left")
    rt.start(None, 0.0)
    thumb = rt.target(with_model(synthetic_angles({}, 0.4, 0.6)))[1]["thumb_1"]
    for curl in (0.3, 0.6, 0.9):
        q = rt.target(with_model(synthetic_angles({"index": curl}, 0.4, 0.6)))[1]
        assert q["thumb_1"] == pytest.approx(thumb, abs=1e-6)
