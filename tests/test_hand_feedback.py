"""RH56F1 sensors -> Nova 2 haptics (10.05 user: the glove limits the operator to
what the 6-joint hand can do and lets him feel contact)."""

from __future__ import annotations

import pytest

from motion_acq.hand.feedback import (
    BRAKES,
    HAPTIC_JOINTS,
    OFF,
    FeedbackConfig,
    HapticFeedback,
    feedback_config,
    palm_force_n,
    tip_forces_n,
)

CLOSED = {"thumb_2": 0.474555, "index_1": 1.5285594, "middle_1": 1.5285594, "ring_1": 1.5285594,
          "pinky_1": 1.5285594}
OPEN_Q = {j: 0.0 for j in CLOSED} | {"thumb_1": 1.57}
NO_TOUCH = [0, 0, 0, 0, 0]  # pinky, ring, middle, index, thumb
PALM_NONE = [0] * 9


def fb(**cfg) -> HapticFeedback:
    return HapticFeedback(FeedbackConfig(**cfg), CLOSED)


def step(f: HapticFeedback, t: float, *, touch=NO_TOUCH, palm=PALM_NONE, q=OPEN_Q, measured=OPEN_Q,
         force=None, active=True):
    f.on_touch(touch, palm, t)
    if force is not None:
        f.on_joint_force([f"r_hj_{j}" for j in force], list(force.values()), t)
    return f.update(t, active=active, q_command=q, q_measured=measured)


def test_free_hand_gives_no_feedback():
    h = step(fb(), 0.0)
    assert h.efforts() == [0.0] * len(HAPTIC_JOINTS)


def test_tip_contact_brakes_that_glove_finger_in_proportion():
    light = step(fb(), 0.0, touch=[0, 0, 0, 100, 0])  # index 1.0 N
    assert 0.2 < light.brake["index"] < 0.4
    assert light.brake["middle"] == light.brake["thumb"] == light.brake["ring"] == 0.0
    firm = step(fb(), 0.0, touch=[0, 0, 0, 400, 0])  # 4 N
    assert firm.brake["index"] == 1.0


def test_pinky_contact_folds_into_the_ring_brake():
    h = step(fb(), 0.0, touch=[200, 0, 0, 0, 0])
    assert h.brake["ring"] > 0.5 and h.vibration["palm_pinky"] > 0.0


def test_unread_tactile_cells_are_no_contact():
    assert tip_forces_n([65535, -1, 0, 50, 0]) == {"pinky": 0.0, "ring": 0.0, "middle": 0.0, "index": 0.5,
                                                   "thumb": 0.0}
    with pytest.raises(ValueError):
        tip_forces_n([0, 0, 0])


def test_held_back_finger_brakes_even_without_tip_contact():
    """A first-phalanx contact has no tip force: the joint force while the command is ahead tells."""
    q = OPEN_Q | {"middle_1": 1.0}
    blocked = OPEN_Q | {"middle_1": 0.6}
    h = step(fb(), 0.0, q=q, measured=blocked, force={"middle_1": 450})
    assert h.brake["middle"] == 1.0


def test_joint_force_in_free_motion_does_not_brake():
    """Lagging behind a fast command with the free-motion force (< 70) is not contact."""
    q = OPEN_Q | {"index_1": 1.0}
    lag = OPEN_Q | {"index_1": 0.6}
    assert step(fb(), 0.0, q=q, measured=lag, force={"index_1": 60}).brake["index"] == 0.0


def test_force_without_a_command_ahead_does_not_brake():
    """Opening against something (command behind position) never locks the operator."""
    q = OPEN_Q | {"index_1": 0.4}
    at = OPEN_Q | {"index_1": 0.6}
    assert step(fb(), 0.0, q=q, measured=at, force={"index_1": 450}).brake["index"] == 0.0


def test_robot_finger_at_its_closed_end_limits_the_glove_finger():
    q = OPEN_Q | {"index_1": CLOSED["index_1"]}
    assert step(fb(), 0.0, q=q, measured=q).brake["index"] == 1.0
    assert step(fb(limit_level=0.0), 0.0, q=q, measured=q).brake["index"] == 0.0


def test_brake_hysteresis():
    f = fb()
    assert step(f, 0.0, touch=[0, 0, 0, 33, 0]).brake["index"] == 0.0  # 0.33 N -> 0.011: below brake_on
    assert step(f, 0.1, touch=[0, 0, 0, 80, 0]).brake["index"] > 0.15  # on
    assert step(f, 0.2, touch=[0, 0, 0, 50, 0]).brake["index"] > 0.0  # 0.074: held above brake_off
    assert step(f, 0.3, touch=[0, 0, 0, 40, 0]).brake["index"] == 0.0  # 0.037: released


def test_first_touch_pulses_once():
    f = fb()
    assert step(f, 0.00, touch=[0, 0, 0, 100, 0]).vibration["index_dip"] == pytest.approx(0.6)
    assert step(f, 0.05, touch=[0, 0, 0, 100, 0]).vibration["index_dip"] == pytest.approx(0.6)
    assert step(f, 0.10, touch=[0, 0, 0, 100, 0]).vibration["index_dip"] == 0.0  # pulse over, still touching
    step(f, 0.20)
    assert step(f, 0.30, touch=[0, 0, 0, 100, 0]).vibration["index_dip"] == pytest.approx(0.6)  # new touch


def test_palm_contact_squeezes_the_strap_capped():
    assert palm_force_n([300, 9, 1, 0, 0, 0, 65535, 0, 0]) == pytest.approx(3.0)
    h = step(fb(), 0.0, palm=[2000, 0, 0, 0, 0, 0, 0, 0, 0])
    assert h.squeeze == pytest.approx(0.5)


def test_nothing_unless_following_the_glove():
    f = fb()
    assert step(f, 0.0, touch=[0, 0, 0, 400, 0], active=False) is OFF


def test_stale_sensor_data_is_no_contact():
    f = fb()
    f.on_touch([0, 0, 0, 400, 0], PALM_NONE, 0.0)
    assert f.update(1.0, active=True, q_command=OPEN_Q, q_measured=OPEN_Q).brake["index"] == 0.0
    assert f.sensors(1.0)["tip_force_n"] is None


def test_efforts_are_percent_in_controller_joint_order():
    h = step(fb(), 0.0, touch=[0, 0, 0, 400, 0])
    efforts = h.efforts()
    assert len(efforts) == len(HAPTIC_JOINTS) == 9
    assert efforts[HAPTIC_JOINTS.index("index_brake")] == 100.0
    assert efforts[HAPTIC_JOINTS.index("index_dip")] == 60.0
    assert all(0.0 <= e <= 100.0 for e in efforts)
    assert tuple(f"{b}_brake" for b in BRAKES) == HAPTIC_JOINTS[:4]


def test_config_checks():
    assert feedback_config(None) == FeedbackConfig()
    with pytest.raises(ValueError, match="unknown"):
        feedback_config({"tip_on": 1})
    with pytest.raises(ValueError):
        feedback_config({"tip_on_n": 3.0, "tip_full_n": 1.0})
    with pytest.raises(ValueError):
        feedback_config({"max_squeeze": 2.0})


def test_heartbeat_changes_only_on_levels():
    """The patched glove driver releases a command unchanged for 1 s: the hand node alternates it."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "ros_ws/src/motion_acq_hand/motion_acq_hand/hand_node.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("def heartbeat(")
    namespace: dict = {}
    exec(source[start:source.index("\n\n\n", start)], namespace)  # pure helper, no ROS imports
    heartbeat = namespace["heartbeat"]
    efforts = [100.0, 0.0, 30.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0]
    assert heartbeat(efforts, False) == efforts
    beat = heartbeat(efforts, True)
    assert beat != efforts and beat[1] == 0.0 and beat[0] == pytest.approx(99.99)
    assert heartbeat([0.0] * 9, True) == [0.0] * 9
