"""Kinematic retargeting (10.06 user: retarget the Nova 2 hand onto the RH56F1)."""

from __future__ import annotations

import numpy as np
import pytest
from hand_fixtures import CONFIG, make_calibration

from motion_acq.hand.calibration import align_open_hand, rezero
from motion_acq.hand.fk import TipFk
from motion_acq.hand.kinematic import KinematicRetargeter, TipTables, glove_points, table_path, umeyama
from motion_acq.hand.retarget import HandRetargeter
from motion_acq.hand.rh56f1 import load_rh56f1_map
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles, synthetic_hand_model

HAND_MAP = load_rh56f1_map()


def with_model(angles, side="left"):
    hand, tips = synthetic_hand_model(angles, side)
    return {**angles, **glove_points(hand, tips)}


def test_umeyama_recovers_a_mirrored_scaled_frame():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(8, 3))
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = rot @ np.diag([-1.0, 1.0, 1.0])  # a left-handed glove frame
    dst = 0.9 * src @ rot.T + np.array([1.0, 2.0, 3.0])
    a = umeyama(src, dst)
    assert a.scale == pytest.approx(0.9) and np.allclose(a.rotation, rot)
    assert np.allclose(a.apply(src[0]), dst[0])


def test_glove_without_hand_model_gives_no_points():
    assert glove_points([(0, 0, 0)] * 20, [(0, 0, 0)] * 5) == {}
    assert glove_points([(1, 2, 3)] * 19, [(1, 2, 3)] * 5) == {}


@pytest.fixture(scope="module")
def retargeter():
    cal = make_calibration("left")
    cal = rezero(cal, [with_model(POSE_ANGLES["open"])] * 10)
    assert cal.alignment is not None and cal.alignment.scale == pytest.approx(1 / 1100, rel=0.02)
    assert np.linalg.det(cal.alignment.rotation) < 0  # the synthetic glove frame is mirrored
    return HandRetargeter(CONFIG, cal, HAND_MAP, "left")


def run(rt, signals, steps=90):
    rt.start(None, 0.0)
    t, step = 0.0, None
    for _ in range(steps):
        t += 1 / 30
        step = rt.step(signals, t)
    return step


@pytest.mark.parametrize("pose,finger", [("pinch_index", "index"), ("pinch_middle", "middle")])
def test_a_pinch_puts_the_robot_tips_together(retargeter, pose, finger):
    angles = synthetic_angles({finger: 0.47}, 0.42 if finger == "index" else 0.37,
                              0.0 if finger == "index" else 0.84)
    angles = with_model(angles)
    step = run(retargeter, angles)
    assert retargeter.method_used == "kinematic"
    urdf = __import__("pathlib").Path.home() / "rl_ws/urdf/vendor/RH56F1/RH56F1_L/urdf/RH56F1_L.urdf"
    if urdf.exists():
        assert TipFk(urdf).tip_distance(step.q_command, finger) < 0.008


@pytest.mark.parametrize("finger,joint", [("index", "index_1"), ("middle", "middle_1"), ("ring", "ring_1")])
def test_one_finger_moves_only_its_robot_finger(retargeter, finger, joint):
    flat = run(retargeter, with_model(POSE_ANGLES["flat"]))
    curled = run(retargeter, with_model(synthetic_angles({finger: 1.0}, 0.0, 0.0)))
    assert curled.q_command[joint] > 1.4
    for other in ("thumb_1", "thumb_2"):
        assert curled.q_command[other] == pytest.approx(flat.q_command[other], abs=0.05)
    for other in {"index_1", "middle_1", "ring_1"} - {joint}:
        assert curled.q_command[other] < 0.05


def test_thumb_rotation_spans_its_range(retargeter):
    seq = [run(retargeter, with_model(synthetic_angles(0.0, 0.0, opp, spread=spread))).q_command["thumb_1"]
           for spread, opp in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))]
    assert seq == sorted(seq) and seq[0] < 1.0 and seq[-1] > 1.9


def test_without_hand_model_data_the_example_map_follows(retargeter):
    step = run(retargeter, POSE_ANGLES["fist"])
    assert retargeter.method_used == "examples" and step.q_command["index_1"] > 1.4


def test_solve_is_fast():
    import time

    t = TipTables.load(table_path("right"))
    kr = KinematicRetargeter(t, umeyama(t.open_points(), t.open_points()))
    pts = {k: v for k, v in zip(
        [f"knuckle_{f}" for f in ("index", "middle", "ring", "pinky")] + [f"tip_{f}" for f in
                                                                         ("thumb", "index", "middle", "ring", "pinky")],
        list(t.open_points()[:4]) + [t.thumb_tips[0]] + list(t.open_points()[4:]))}
    start = time.perf_counter()
    for _ in range(50):
        kr.solve(pts)
    assert (time.perf_counter() - start) / 50 < 0.004  # well inside the 8.3 ms tick
    assert align_open_hand("right", []) is None


def test_flat_hand_alignment_needs_the_thumb_to_pick_the_right_handedness():
    """10.06 left: the eight finger points of a flat hand are near one plane, so a mirror fits them
    as well; one real fit came out mirrored (operator thumb 11 cm off, thumb pinned)."""
    from motion_acq.hand.kinematic import FINGERS, fit_alignment, knuckle_prefix, tip_prefix

    t = TipTables.load(table_path("left"))
    robot = t.open_points().copy()
    plane = robot[:, 2].mean()
    robot[:, 2] = plane  # exactly planar fingers: both handedness fit them
    thumb = t.thumb_tip((1.57, 0.0))
    mirror = np.diag([-1100.0, 1100.0, 1100.0])  # left-handed glove frame, mm

    def to_glove(x):
        return mirror @ x + np.array([12.0, -30.0, -150.0])

    pts = {}
    for k, f in enumerate(FINGERS):
        pts[knuckle_prefix(f)] = to_glove(robot[k])
        pts[tip_prefix(f)] = to_glove(robot[4 + k])
    pts[tip_prefix("thumb")] = to_glove(thumb)
    with_thumb = fit_alignment(t, pts, (1.57, 0.0))
    without = fit_alignment(t, pts)
    gap_with = np.linalg.norm(with_thumb.apply(pts[tip_prefix("thumb")]) - thumb)
    gap_without = np.linalg.norm(without.apply(pts[tip_prefix("thumb")]) - thumb)
    assert np.linalg.det(with_thumb.rotation) < 0 and gap_with < 0.015
    assert gap_without > 3 * gap_with  # (squashed fingers fit the table only roughly)


def test_thumb_bend_prior_moves_thumb_2_with_the_operator_thumb():
    """10.06 left: tip matching sent the operator's thumb bend to thumb_1; thumb_2 sat near 0."""
    import dataclasses

    from motion_acq.hand.kinematic import KinematicConfig, points_from_signals

    cal = rezero(make_calibration("left"), [with_model(POSE_ANGLES["open"])] * 10)
    t = TipTables.load(table_path("left"))
    bent = with_model(synthetic_angles(0.0, 1.0, 0.0))
    assert cal.thumb_bend_ratio(bent) == pytest.approx(1.0, abs=0.05)
    assert cal.thumb_bend_ratio(with_model(POSE_ANGLES["flat"])) == pytest.approx(0.0, abs=0.05)
    pts = points_from_signals(bent)
    off = KinematicRetargeter(t, cal.alignment, dataclasses.replace(KinematicConfig(), thumb_bend_weight=0.0))
    on = KinematicRetargeter(t, cal.alignment, dataclasses.replace(KinematicConfig(), thumb_bend_weight=0.03))
    # the synthetic hand model already puts the tip where the robot's bent thumb is: both find it,
    # the prior must not pull it away
    assert on.solve(pts, 0.4746)["thumb_2"] >= off.solve(pts, 0.4746)["thumb_2"] - 0.02
    # a tip left at the straight-thumb position but a bent glove joint: the prior bends thumb_2
    straight = points_from_signals(with_model(POSE_ANGLES["flat"]))
    assert on.solve(straight, 0.4746)["thumb_2"] > off.solve(straight, 0.4746)["thumb_2"] + 0.05
