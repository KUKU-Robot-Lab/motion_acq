"""STEP 3 hand retargeting: glove parsing, calibration, RH56F1 mapping."""

from __future__ import annotations

import math

import pytest

from hand_fixtures import CONFIG, make_calibration

from motion_acq.hand.calibration import CalibrationError, HandCalibration, rezero
from motion_acq.hand.nova2 import (
    GloveDataError,
    angles_from_state,
    glove_joint_names,
    tip_distances,
)
from motion_acq.hand.retarget import (
    HandRetargeter,
    HandState,
)
from motion_acq.hand.rh56f1 import LEAVE, load_rh56f1_map
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles, synthetic_state

HAND_MAP = load_rh56f1_map()


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


def test_calibration_roundtrip_and_side(tmp_path):
    cal = make_calibration("left")
    path = tmp_path / "op_left.yaml"
    cal.save(path)
    loaded = HandCalibration.load(path, side="left")
    probe = POSE_ANGLES["pinch_middle"]
    assert loaded.predict(probe) == pytest.approx(cal.predict(probe), abs=1e-9)
    with pytest.raises(CalibrationError, match="left hand calibration"):
        HandCalibration.load(path, side="right")


def test_old_open_fist_calibration_asks_for_a_new_one(tmp_path):
    path = tmp_path / "op_right.yaml"
    path.write_text("schema: motion_acq/hand_calibration/v1\nside: right\nranges: {index: {open: 0, closed: 1}}\n")
    with pytest.raises(CalibrationError, match="calibrate again"):
        HandCalibration.load(path, side="right")


def test_a_broken_calibration_file_is_rejected(tmp_path):
    cal = make_calibration("right")
    raw = cal.to_dict()
    raw["model"]["std"][0] = 0.0
    path = tmp_path / "bad.yaml"
    import yaml
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(CalibrationError, match="zero spread"):
        HandCalibration.load(path)


def make_retargeter(side: str = "right") -> HandRetargeter:
    return HandRetargeter(CONFIG, make_calibration(side), HAND_MAP, side)


def test_idle_until_started():
    rt = make_retargeter()
    step = rt.step(POSE_ANGLES["fist"], 0.1)
    assert step.state is HandState.IDLE and step.registers is None


@pytest.mark.parametrize("pose", sorted(CONFIG.examples))
def test_every_example_pose_reaches_its_robot_pose(pose):
    rt = make_retargeter()
    rt.start(None, 0.0)
    _, step = run(rt, POSE_ANGLES[pose], 120)
    assert step.q_command == pytest.approx(dict(CONFIG.examples[pose].target), abs=0.03)


def test_fist_reaches_the_vendor_curl_registers():
    rt = make_retargeter()
    rt.start(None, 0.0)
    _, step = run(rt, POSE_ANGLES["fist"], 120)
    # Right-hand calibration (sim2real 09.30 sweep) puts full curl at 926/920/900/905.
    assert step.registers == HAND_MAP.to_registers(step.q_command, side="right")
    assert step.registers[:4] == [926, 920, 900, 905]


def test_beyond_the_examples_is_clamped_to_the_joint_limits():
    rt = make_retargeter()
    rt.start(None, 0.0)
    _, step = run(rt, synthetic_angles(1.6, 1.6, 1.6), 120)
    for j, (lo, hi) in CONFIG.limits_rad.items():
        assert lo - 1e-9 <= step.q_command[j] <= hi + 1e-9


@pytest.mark.parametrize("finger,joint", [("index", "index_1"), ("middle", "middle_1"), ("ring", "ring_1")])
def test_one_finger_moves_only_its_robot_finger(finger, joint):
    """10.05 user: moving one finger moved the robot thumb too (coupled glove signals: ring curl
    reads as thumb rotation, a curling finger nears the thumb)."""
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, flat = run(rt, POSE_ANGLES["flat"], 60)
    for amount in (0.3, 0.6, 1.0):
        t, step = run(rt, synthetic_angles({finger: amount}, 0.0, 0.0), 60, t)
        assert step.q_command[joint] > 0.4 * amount
        for other in ("thumb_1", "thumb_2"):
            assert step.q_command[other] == pytest.approx(flat.q_command[other], abs=0.08), (amount, other)
        for other in {"index_1", "middle_1", "ring_1"} - {joint}:
            assert step.q_command[other] < 0.1, (amount, other)


def test_in_between_poses_interpolate_smoothly():
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["flat"], 30)
    values = []
    for amount in (0.0, 0.25, 0.5, 0.75, 1.0):
        t, step = run(rt, synthetic_angles({"index": amount}, 0.0, 0.0), 60, t)
        values.append(step.q_command["index_1"])
    assert values == sorted(values) and values[-1] - values[0] > 1.3


def test_a_pinch_approach_ends_at_the_tips_touching():
    rt = make_retargeter()
    rt.start(None, 0.0)
    t, _ = run(rt, POSE_ANGLES["flat"], 30)
    target = CONFIG.examples["pinch_index"].target
    gaps = []
    for amount in (0.0, 0.5, 1.0):
        t, step = run(rt, synthetic_angles({"index": 0.45}, 0.5, 0.35, ("index", amount)), 90, t)
        gaps.append(max(abs(step.q_command[j] - target[j]) for j in ("thumb_1", "thumb_2", "index_1")))
    assert gaps == sorted(gaps, reverse=True) and gaps[-1] < 0.03


def test_without_tip_data_the_joints_still_follow():
    rt = make_retargeter()
    rt.start(None, 0.0)
    no_tips = {k: v for k, v in POSE_ANGLES["fist"].items() if not k.startswith("tipdist")}
    _, step = run(rt, no_tips, 120)
    assert step.q_command["index_1"] > 1.2
    assert set(rt.missing_inputs) == {"tipdist_index", "tipdist_middle", "tipdist_ring"}


def test_rezero_follows_a_shifted_glove_without_redoing_the_examples():
    """After a SenseCom restart the readings shift (bumsu 09-22: fingers stayed bent): the
    2 s open hand shifts the inputs back; the examples are not redone."""
    cal = make_calibration()
    shift = {k: (0.3 if k.endswith(("_mcp", "_pip", "_dip")) else 0.0) for k in POSE_ANGLES["open"]}
    shifted = {p: {k: v + shift[k] for k, v in a.items()} for p, a in POSE_ANGLES.items()}
    assert cal.predict(shifted["open"])["index_1"] > 0.15  # off before
    fixed = rezero(cal, [shifted["open"]] * 20)
    for pose in ("open", "fist", "index", "pinch_middle"):
        q = fixed.predict(shifted[pose])
        assert q == pytest.approx(dict(CONFIG.examples[pose].target), abs=0.05), pose
    assert fixed.rezeroed is not None


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


def test_tip_distances_from_the_glove():
    tips = [(0, 0, 0), (30, 40, 0), (0, 0, 50), (60, 80, 0), (1, 0, 0)]
    d = tip_distances(tips)
    assert d["tipdist_index"] == pytest.approx(50.0) and d["tipdist_ring"] == pytest.approx(100.0)
    assert tip_distances([(0, 0, 0)] * 5) == {}  # not filled
    assert tip_distances(tips[:3]) == {}


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
        for pose, example in CONFIG.examples.items():
            if pose.startswith("pinch_"):
                assert fk.tip_distance(example.target, pose.removeprefix("pinch_")) < 0.012, (side, pose)
