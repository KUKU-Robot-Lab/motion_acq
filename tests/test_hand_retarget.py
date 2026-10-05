"""STEP 3 hand retargeting: glove parsing, calibration, RH56F1 mapping."""

from __future__ import annotations

import math

import pytest

from motion_acq.hand.calibration import CalibrationError, HandCalibration, calibrate
from motion_acq.hand.nova2 import (
    GloveDataError,
    angles_from_state,
    features,
    glove_joint_names,
)
from motion_acq.hand.retarget import (
    DEFAULT_RETARGET,
    HandRetargeter,
    HandState,
    load_hand_retarget_config,
)
from motion_acq.hand.rh56f1 import LEAVE, load_rh56f1_map
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles, synthetic_state

CONFIG = load_hand_retarget_config()
HAND_MAP = load_rh56f1_map()


def make_calibration(side: str = "right") -> HandCalibration:
    samples = {pose: [features(a, CONFIG.features)] * 5 for pose, a in POSE_ANGLES.items()}
    return calibrate(side=side, user="test", pose_samples=samples,
                     feature_poses=CONFIG.feature_poses, min_span=CONFIG.min_span_rad)


def run(rt: HandRetargeter, angles, steps: int, t0: float = 0.0, dt: float = 1 / 30):
    t, step = t0, None
    for _ in range(steps):
        t += dt
        step = rt.step(angles, t)
    return t, step


def test_rh56f1_home_registers_regression():
    # Cross-checked against sim2real policy_control/rh56f1_map.py (6000 samples identical).
    assert HAND_MAP.to_registers(CONFIG.home_rad, side="right") == [1740, 1740, 1740, 1740, 1350, 853]
    assert HAND_MAP.to_registers(CONFIG.home_rad, side="left") == [1740, 1740, 1740, 1740, 1350, 916]
    assert HAND_MAP.to_registers(CONFIG.home_rad, side=None)[5] == 1750 - round(1.57 / 2.0943951 * 1150)


def test_registers_are_clipped_to_the_vendor_range():
    q = {name: 10.0 for name in HAND_MAP.joint_order}
    regs = HAND_MAP.to_registers(q, side="right")
    assert regs[:4] == [900] * 4 and regs[4] == 1100 and regs[5] == 600
    with pytest.raises(ValueError):
        HAND_MAP.to_registers({**q, "index_1": math.nan}, side="right")


def test_glove_state_parsing():
    names, positions = synthetic_state("left", POSE_ANGLES["fist"])
    assert tuple(names) == glove_joint_names("left") and len(names) == 20
    angles = angles_from_state(names, positions, "left")
    assert angles["index_pip"] == pytest.approx(1.74)
    with pytest.raises(GloveDataError, match="not a right"):
        angles_from_state(names, positions, "right")
    with pytest.raises(GloveDataError, match="missing"):
        angles_from_state(names[:-1], positions[:-1], "left")
    with pytest.raises(GloveDataError, match="non-finite"):
        angles_from_state(names, [math.nan] + positions[1:], "left")


def test_calibration_ranges_and_roundtrip(tmp_path):
    cal = make_calibration("left")
    assert set(cal.ranges) == {s.name for s in CONFIG.features}
    path = tmp_path / "op_left.yaml"
    cal.save(path)
    loaded = HandCalibration.load(path, side="left")
    assert loaded.ranges == cal.ranges
    with pytest.raises(CalibrationError, match="left hand calibration"):
        HandCalibration.load(path, side="right")


def test_calibration_rejects_indistinct_poses():
    samples = {pose: [features(POSE_ANGLES["open"], CONFIG.features)] for pose in POSE_ANGLES}
    with pytest.raises(CalibrationError, match="redo those poses"):
        calibrate(side="right", user="t", pose_samples=samples,
                  feature_poses=CONFIG.feature_poses, min_span=CONFIG.min_span_rad)


def test_calibration_requires_every_pose():
    samples = {"open": [features(POSE_ANGLES["open"], CONFIG.features)]}
    with pytest.raises(CalibrationError, match="not recorded"):
        calibrate(side="right", user="t", pose_samples=samples,
                  feature_poses=CONFIG.feature_poses, min_span=CONFIG.min_span_rad)


def make_retargeter(side: str = "right") -> HandRetargeter:
    return HandRetargeter(CONFIG, make_calibration(side), HAND_MAP, side)


def test_idle_until_started():
    rt = make_retargeter()
    step = rt.step(POSE_ANGLES["fist"], 0.1)
    assert step.state is HandState.IDLE and step.registers is None


def test_open_fist_and_opposition_map_to_rh56f1_ends():
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, step = run(rt, POSE_ANGLES["open"], 60)
    # open hand: fingers and thumb bend at home, the spread thumb out to 0.8 (10.05: full rotation range)
    assert step.q_command == pytest.approx({**CONFIG.home_rad, "thumb_1": 0.8}, abs=1e-3)
    t, step = run(rt, POSE_ANGLES["fist"], 90, t)
    assert step.q_command["index_1"] == pytest.approx(1.5285594, abs=1e-3)
    assert step.q_command["thumb_2"] == pytest.approx(0.474555, abs=1e-3)
    assert step.q_command["thumb_1"] == pytest.approx(0.8, abs=1e-3)
    assert rt.pinch_weight == 0.0  # the thumb near the fingers in a fist is not a pinch
    # Right-hand calibration (sim2real 09.30 sweep) puts full curl at 926/920/900/905.
    assert step.registers == HAND_MAP.to_registers(step.q_command, side="right")
    assert step.registers[:4] == [926, 920, 900, 905]
    _, step = run(rt, POSE_ANGLES["thumb_opposed"], 90, t)
    assert step.q_command["thumb_1"] == pytest.approx(2.0, abs=1e-3)
    assert step.q_command["index_1"] == pytest.approx(0.0, abs=1e-3)


def test_beyond_calibrated_poses_is_clamped():
    rt = make_retargeter()
    rt.start(None, 0.0)
    _, step = run(rt, synthetic_angles(1.5, 1.5, 1.5), 120)
    assert all(0.0 <= n <= 1.0 for n in step.normalized.values())
    assert step.q_command["index_1"] == pytest.approx(1.5285594, abs=1e-3)


def test_rate_limit_per_joint():
    rt = make_retargeter()
    rt.start(None, 0.0)
    dt, t, prev = 1 / 30, 0.0, rt.command()
    worst = 0.0
    for _ in range(60):
        t += dt
        step = rt.step(POSE_ANGLES["fist"], t)
        worst = max(worst, max(abs(step.q_command[j] - prev[j]) / dt for j in prev))
        prev = dict(step.q_command)
    assert worst <= CONFIG.max_velocity_rad_s + 1e-6


def test_glove_loss_holds_last_registers():
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, step = run(rt, synthetic_angles(0.5, 0.5, 0.5), 60)
    held, held_q = list(step.registers), dict(step.q_command)
    for _ in range(5):
        t += 1 / 30
        hold = rt.step(None, t)
        assert hold.state is HandState.HOLD and hold.registers == held
        assert hold.q_command == held_q
    _, resumed = run(rt, POSE_ANGLES["fist"], 1, t)
    assert resumed.state is HandState.RUNNING
    # Velocity restarts from zero after HOLD: the first step moves at most a*dt^2.
    limit = CONFIG.max_acceleration_rad_s2 * (1 / 30) ** 2 + 1e-9
    assert all(abs(resumed.q_command[j] - held_q[j]) <= limit for j in held_q)


def test_start_from_measured_pose_has_no_jump():
    rt = make_retargeter()
    measured = {**CONFIG.home_rad, "index_1": 1.0, "thumb_1": 1.2}
    rt.start(measured, 0.0)
    assert rt.command()["index_1"] == pytest.approx(1.0)
    _, step = run(rt, POSE_ANGLES["open"], 1)
    assert abs(step.q_command["index_1"] - 1.0) <= CONFIG.max_acceleration_rad_s2 * (1 / 30) ** 2 + 1e-9


def test_side_mismatch_and_unverified_axes():
    with pytest.raises(ValueError, match="calibration is for"):
        HandRetargeter(CONFIG, make_calibration("left"), HAND_MAP, "right")
    import dataclasses
    axes = tuple(dataclasses.replace(a, verified=a.name != "thumb_1") for a in HAND_MAP.axes)
    partial = dataclasses.replace(HAND_MAP, axes=axes, side_axes={})
    assert partial.to_registers(CONFIG.home_rad, side=None)[5] == LEAVE


# -- Nova 2 gloves: serial and inventory -----------------------------------------

def test_glove_serial_keeps_leading_zeros_and_rejects_numbers():
    from motion_acq.hand.nova2 import check_serial, glove_topic

    assert glove_topic("00782", "right") == "/senseglove/glove00782/rh/senseglove_states"
    assert glove_topic("0", "left") == "/senseglove/glove0/lh/senseglove_states"
    # what ROS (-p glove_serial:=00782) and unquoted YAML turn the serial into
    for bad in (782.0, 782, "782.0", "", None, "00 782"):
        with pytest.raises(GloveDataError, match="quoted digit string"):
            check_serial(bad)


def test_lab_glove_inventory():
    from motion_acq.hand.nova2 import load_gloves

    gloves = load_gloves()
    assert set(gloves) == {"right", "left"}
    assert gloves["right"].serial == "00782" and gloves["left"].serial == "00795"
    assert gloves["right"].topic == "/senseglove/glove00782/rh/senseglove_states"
    assert gloves["left"].name == "Nova 2-00795-L"


def test_glove_inventory_validation(tmp_path):
    from motion_acq.hand.nova2 import load_gloves

    path = tmp_path / "gloves.yaml"
    # unquoted 01001 is YAML 1.1 octal (513); unquoted 00782 happens to stay text
    path.write_text('gloves:\n  right: {serial: 01001, mac: "E8:6B:EA:C8:16:B2", name: "Nova 2-01001-R"}\n')
    with pytest.raises(GloveDataError, match="quoted"):
        load_gloves(path)
    path.write_text('gloves:\n  right: {serial: "00782", mac: "E8:6B:EA:C8:16", name: "Nova 2-00782-R"}\n')
    with pytest.raises(GloveDataError, match="mac"):
        load_gloves(path)
    path.write_text('gloves:\n  right: {serial: "00782", mac: "E8:6B:EA:C8:16:B2", name: "Nova 2-00795-L"}\n')
    with pytest.raises(GloveDataError, match="name"):
        load_gloves(path)


def test_features_follow_the_glove_joints_that_move():
    """mcp + pip per finger (10.05: a base-only bend was lost), pinky = ring (no sensor), thumb_brake rotates."""
    assert {s.name: dict(s.weights) for s in CONFIG.features} == {
        "index": {"index_mcp": 1.0, "index_pip": 1.0}, "middle": {"middle_mcp": 1.0, "middle_pip": 1.0},
        "ring": {"ring_mcp": 1.0, "ring_pip": 1.0}, "pinky": {"ring_mcp": 1.0, "ring_pip": 1.0},
        "thumb_bend": {"thumb_pip": 1.0},
        "pinch_index": {"tipdist_index": 1.0}, "pinch_middle": {"tipdist_middle": 1.0},
        "pinch_ring": {"tipdist_ring": 1.0},
        "thumb_opposition": {"thumb_brake": 1.0},
    }
    cal = make_calibration()
    assert cal.ranges["thumb_opposition"].closed > cal.ranges["thumb_opposition"].open


def test_a_base_only_bend_moves_the_robot_finger():
    """10.05 user: bending a finger only at its base (mcp) barely moved the robot finger."""
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["open"], 30)
    base = dict(POSE_ANGLES["open"], index_mcp=POSE_ANGLES["fist"]["index_mcp"])
    _, step = run(rt, base, 60, t)
    assert step.q_command["index_1"] > 0.5
    assert step.q_command["middle_1"] == pytest.approx(0.0, abs=1e-3)


def test_a_working_grip_short_of_the_calibration_fist_closes_fully():
    """10.05 user: the power grip reached only 85-89 % of the calibration fist."""
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["open"], 30)
    short = synthetic_angles(0.87, 0.87, 0.0)
    _, step = run(rt, short, 90, t)
    assert step.q_command["index_1"] == pytest.approx(1.5285594, abs=1e-3)
    assert step.q_command["thumb_2"] == pytest.approx(0.474555, abs=1e-3)


@pytest.mark.parametrize("side", ["right", "left"])
def test_closing_the_thumb_sends_it_across_the_palm(side):
    """10.05, right hand: register 852 -> 620 swings the thumb across the palm, 852 -> 1138 away."""
    rt = HandRetargeter(CONFIG, make_calibration(side), HAND_MAP, side)
    rt.start(None, 0.0)
    t, home = run(rt, POSE_ANGLES["open"], 30)
    _, opposed = run(rt, POSE_ANGLES["thumb_opposed"], 90, t)
    assert opposed.registers[5] < home.registers[5] - 400
    assert opposed.registers[5] >= 600


def test_end_margins_are_checked(tmp_path):
    import yaml

    raw = yaml.safe_load(DEFAULT_RETARGET.read_text(encoding="utf-8"))
    raw["end_margins"] = {"open": 0.5, "closed": 0.5}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="end_margins"):
        load_hand_retarget_config(bad)


@pytest.mark.parametrize("pose,finger", [("pinch_index", "index_1"), ("pinch_middle", "middle_1"),
                                         ("pinch_ring", "ring_1")])
def test_a_pinch_puts_the_robot_tips_together(pose, finger):
    """10.05 user: the thumb must be able to meet the other fingers. The calibrated touch
    sends the joints to the FK pose where the RH56F1 tips meet."""
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["open"], 30)
    _, step = run(rt, POSE_ANGLES[pose], 120, t)
    target = CONFIG.pinch_targets[pose]
    assert rt.pinch == pose and rt.pinch_weight == pytest.approx(1.0)
    for joint in ("thumb_1", "thumb_2", finger):
        assert step.q_command[joint] == pytest.approx(target[joint], abs=1e-3)


def test_pinch_blends_in_smoothly_as_the_thumb_nears():
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["open"], 30)
    weights = []
    for amount in (0.0, 0.3, 0.55, 0.7, 0.85, 1.0):
        t, _ = run(rt, synthetic_angles(0.45, 0.4, 0.3, ("index", amount)), 20, t)
        weights.append(rt.pinch_weight if rt.pinch == "pinch_index" else 0.0)
    assert weights[0] == 0.0 and weights[-1] == pytest.approx(1.0)
    assert weights == sorted(weights) and 0.0 < weights[3] < 1.0


def test_without_tip_data_the_joints_still_follow():
    rt = make_retargeter()
    rt.start(None, 0.0)
    no_tips = {k: v for k, v in POSE_ANGLES["fist"].items() if not k.startswith("tipdist")}
    _, step = run(rt, no_tips, 90)
    assert step.q_command["index_1"] == pytest.approx(1.5285594, abs=1e-3)
    assert rt.pinch is None


def test_pinch_targets_are_where_the_rh56f1_tips_meet():
    """Guard against editing the pinch poses by hand: vendor URDF FK, tips within 12 mm."""
    from pathlib import Path

    from motion_acq.hand.fk import TipFk

    for side in "RL":
        urdf = Path.home() / f"rl_ws/urdf/vendor/RH56F1/RH56F1_{side}/urdf/RH56F1_{side}.urdf"
        if not urdf.exists():
            pytest.skip("vendor URDF not on this host")
        fk = TipFk(urdf)
        assert fk.tip_distance(CONFIG.home_rad, "index") > 0.08  # open hand: tips far apart
        for pose, target in CONFIG.pinch_targets.items():
            assert fk.tip_distance(target, pose.removeprefix("pinch_")) < 0.012, (side, pose)


def test_curling_one_finger_toward_a_still_thumb_does_not_move_the_robot_thumb():
    """10.05 user: moving one finger moved the robot thumb. Its tip came near the still thumb
    (pinch distance), which pulled the thumb into the pinch pose."""
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, before = run(rt, POSE_ANGLES["open"], 30)
    curled = dict(POSE_ANGLES["open"], index_mcp=0.7, index_pip=0.9, index_dip=0.45,
                  tipdist_index=POSE_ANGLES["pinch_index"]["tipdist_index"])
    _, step = run(rt, curled, 90, t)
    assert rt.pinch_weight == 0.0
    assert step.q_command["thumb_1"] == pytest.approx(before.q_command["thumb_1"], abs=1e-3)
    assert step.q_command["thumb_2"] == pytest.approx(before.q_command["thumb_2"], abs=1e-3)
    assert step.q_command["index_1"] > 0.5
