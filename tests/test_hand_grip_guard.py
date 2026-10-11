"""Grip guard (10.06): a finger blocked by an object is not driven further into it.

10.06 left cup grasp: the glove asked for a fist, the blocked fingers kept pushing (1346-1715 g,
6.5 A in total) and the hand froze locked closed 0.15 s after contact.
"""

from __future__ import annotations

import pytest
from test_hand_controller import HAND_MAP, Rig, make_controller, run_until

from motion_acq.hand.grip_guard import GripGuard, GripGuardConfig, grip_guard_config
from motion_acq.hand.synthetic import POSE_ANGLES

NAMES = ["l_hj_pinky_1", "l_hj_ring_1", "l_hj_middle_1", "l_hj_index_1", "l_hj_thumb_2", "l_hj_thumb_1"]
MEASURED = {"thumb_1": 1.2, "thumb_2": 0.2, "index_1": 0.8, "middle_1": 0.8, "ring_1": 0.8, "pinky_1": 0.8}


def guard_with(force=(0, 0, 0, 0, 0, 0), current=(60, 60, 60, 60, 60, 60), t=1.0, **cfg) -> GripGuard:
    g = GripGuard(GripGuardConfig(**cfg))
    g.on_force(NAMES, force, t)
    g.on_current(NAMES, current, t)
    return g


def test_free_motion_is_not_limited():
    """10.06 free motion: <= 320 mA per actuator and < 250 g."""
    assert guard_with(force=(-12, -12, -12, 96, -4, 48), current=(160, 130, 126, 135, 101, 314)).ceilings(MEASURED, 1.0) == {}


def test_a_pressing_finger_closes_no_further_than_where_it_is():
    g = guard_with(force=(0, 0, 0, 500, 0, 0))
    assert g.ceilings(MEASURED, 1.0) == {"index_1": pytest.approx(0.8 + 0.03)}
    assert g.engaged == {"index_1": "hold"}


def test_high_current_alone_engages():
    assert guard_with(current=(60, 60, 900, 60, 60, 60)).ceilings(MEASURED, 1.0) == {"middle_1": pytest.approx(0.83)}


def test_the_cup_grasp_backs_off_every_loaded_joint():
    """55.68 s of the 10.06 bag: force and current just before the hand froze."""
    g = guard_with(force=(1254, 1342, 1328, 1279, 1038, 717), current=(1141, 1037, 1178, 1084, 466, 747))
    ceilings = g.ceilings(MEASURED, 1.0)
    assert set(ceilings) == set(MEASURED)
    assert all(ceilings[j] == pytest.approx(MEASURED[j] - 0.03) for j in MEASURED)


def test_total_current_backs_off_joints_that_are_only_holding():
    g = guard_with(force=(300, 300, 300, 300, 300, 0), current=(700, 700, 700, 700, 700, 60))
    ceilings = g.ceilings(MEASURED, 1.0)  # 3500 mA in total
    assert ceilings["index_1"] == pytest.approx(0.77) and "thumb_1" not in ceilings


def test_hysteresis_keeps_the_hold_until_the_finger_lets_go():
    g = guard_with(force=(0, 0, 0, 500, 0, 0))
    g.ceilings(MEASURED, 1.0)
    g.on_force(NAMES, (0, 0, 0, 300, 0, 0), 1.01)
    assert "index_1" in g.ceilings(MEASURED, 1.01)  # still pressing above force_off_g
    g.on_force(NAMES, (0, 0, 0, 200, 0, 0), 1.02)
    assert g.ceilings(MEASURED, 1.02) == {}


def test_stale_readings_or_disabled_guard_do_nothing():
    g = guard_with(force=(0, 0, 0, 900, 0, 0), t=0.0)
    assert g.ceilings(MEASURED, 1.0) == {}
    assert guard_with(force=(0, 0, 0, 900, 0, 0), enabled=False).ceilings(MEASURED, 1.0) == {}


def test_config_rejects_unknown_keys_and_bad_order():
    with pytest.raises(ValueError, match="unknown"):
        grip_guard_config({"force_on": 1})
    with pytest.raises(ValueError):
        grip_guard_config({"force_on_g": 100.0})  # below force_off_g
    assert grip_guard_config({"enabled": False, "margin_rad": 0.05}).margin_rad == 0.05


# -- in the controller ------------------------------------------------------------------------------

def press_index(rig, grams: int) -> None:
    rig.ctl.on_joint_force([n.replace("l_", "r_") for n in NAMES], [10, 10, 10, grams + len(rig.sent) % 2, 10, 10], rig.t)
    rig.ctl.on_current([n.replace("l_", "r_") for n in NAMES], [60, 60, 60, 60 + len(rig.sent) % 2, 60, 60], rig.t)


def follow_fist_on_a_cup(rig, grams_at_contact: int, steps: int = 40):
    """The index stops on an object at its current register; the glove keeps asking for a fist."""
    rig.ctl.request_enable(True)
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["state"] == "running")
    out = run_until(rig, POSE_ANGLES["fist"], lambda o: o.angle[3] < 1400)
    cup = out.angle[3]
    outs = []
    for _ in range(steps):
        press_index(rig, grams_at_contact)
        out = rig.step(POSE_ANGLES["fist"])
        rig.hand[3] = cup  # blocked
        outs.append(out)
    return cup, outs


def test_controller_stops_closing_a_blocked_finger_but_others_follow():
    """Above force_on_g (950 g in the config: the admittance ceiling is ~800 g) the guard holds the finger."""
    rig = Rig(make_controller())
    cup, outs = follow_fist_on_a_cup(rig, 1000)
    last = outs[-1]
    assert last.record["grip_guard"] == {"index_1": "hold"}
    margin_reg = abs(HAND_MAP.to_registers({**last.record["measured_rad"], "index_1": last.record["measured_rad"]["index_1"] + 0.03},
                                           side="right")[3] - cup)
    assert last.angle[3] >= cup - margin_reg - 1  # registers fall as the finger closes
    assert last.angle[2] < 1000  # the middle finger, not pressing, reaches the fist


def test_controller_opens_at_once_when_the_operator_lets_go():
    rig = Rig(make_controller())
    cup, _ = follow_fist_on_a_cup(rig, 500)
    press_index(rig, 500)
    out = rig.step(POSE_ANGLES["flat"])
    for _ in range(30):
        press_index(rig, 100)
        out = rig.step(POSE_ANGLES["flat"])
    assert out.record["grip_guard"] is None
    assert out.angle[3] > cup + 200  # opened well past the cup


# -- contact speed and force bias (arXiv 2603.08988: overshoot grows with contact speed) ------------

def test_contact_slows_only_that_joint_and_releases_with_hysteresis():
    g = guard_with(force=(0, 0, 0, 150, 0, 0))
    speeds = g.speeds(1.0, 2000)
    assert speeds["index_1"] == 300 and speeds["middle_1"] == 2000
    g.on_force(NAMES, (0, 0, 0, 80, 0, 0), 1.01)
    assert g.speeds(1.01, 2000)["index_1"] == 300  # still above half the contact level
    g.on_force(NAMES, (0, 0, 0, 40, 0, 0), 1.02)
    assert g.speeds(1.02, 2000)["index_1"] == 2000


def test_tip_touch_or_current_also_mean_contact():
    g = guard_with(current=(60, 60, 500, 60, 60, 60))
    g.on_tips({"thumb": 0.3, "index": 0.0}, 1.0)
    speeds = g.speeds(1.0, 2000)
    assert speeds["middle_1"] == speeds["thumb_1"] == speeds["thumb_2"] == 300 and speeds["index_1"] == 2000


def test_rest_bias_is_learnt_and_subtracted():
    """Index reads ~96 g at rest (10.06): 96 + 150 must not count as 246 g pressing."""
    g = guard_with(force=(0, 0, 0, 96, 0, 0))
    for k in range(60):
        g.on_force(NAMES, (0, 0, 0, 96 + k % 3, 0, 0), 1.0)
        g.learn_rest(1.0)
    assert g.bias["index_1"] == pytest.approx(97, abs=1)
    g.on_force(NAMES, (0, 0, 0, 96 + 100, 0, 0), 1.0)
    assert g.speeds(1.0, 2000)["index_1"] == 2000  # 100 g over rest: below contact_force_g
    g.on_force(NAMES, (0, 0, 0, 96 + 450, 0, 0), 1.0)
    assert g.ceilings(MEASURED, 1.0) == {"index_1": pytest.approx(0.83)}


def test_controller_sends_contact_speed_once_stalled_and_driver_speed_after():
    rig = Rig(make_controller())
    follow_fist_on_a_cup(rig, 500, steps=20)
    index_slot = 3
    slow = [i for i, o in enumerate(rig.sent[-20:]) if o.speed is not None and o.speed[index_slot] == 300]
    assert slow and slow[0] <= 15  # within contact_stall_s (0.1 s = 12 ticks) of the block
    assert rig.sent[-20 + slow[0]].speed[2] == 2000  # only the pressing finger slows
    for _ in range(30):
        press_index(rig, 0)
        out = rig.step(POSE_ANGLES["flat"])
    speeds = [o.speed for o in rig.sent[-30:] if o.speed is not None]
    assert speeds[-1] == [2000] * 6
    assert out.record["contact_slow"] is None


# -- contact cap (10.11 arm4090 left cup: the finger crept on for 1.4 s after the operator stopped) --

def test_contact_cap_holds_a_touching_joint_near_where_it_touched():
    g = guard_with(force=(0, 0, 0, 150, 0, 0), contact_cap_rad=0.07)
    ask = {**MEASURED, "index_1": 1.2}
    assert g.contact_ceilings(MEASURED, ask, 1.0) == {"index_1": pytest.approx(0.87)}
    later = {**MEASURED, "index_1": 0.85}  # it moved on: the touch point stays where it touched
    assert g.contact_ceilings(later, ask, 1.1) == {"index_1": pytest.approx(0.87)}


def test_contact_cap_is_kept_while_the_held_finger_reads_little_force():
    """10.11: at its cap the ring read < 60 g; a force-based release re-took the cap deeper (creep)."""
    g = guard_with(force=(0, 0, 0, 150, 0, 0), contact_cap_rad=0.07)
    ask = {**MEASURED, "index_1": 1.2}
    g.contact_ceilings(MEASURED, ask, 1.0)
    g.on_force(NAMES, (0, 0, 0, 30, 0, 0), 2.0)
    deeper = {**MEASURED, "index_1": 0.87}
    assert g.contact_ceilings(deeper, ask, 2.0) == {"index_1": pytest.approx(0.87)}  # same touch, 1 s later
    g.on_force(NAMES, (0, 0, 0, 200, 0, 0), 2.1)
    assert g.contact_ceilings(deeper, ask, 2.1) == {"index_1": pytest.approx(0.87)}


def test_contact_cap_is_released_when_the_operator_opens():
    g = guard_with(force=(0, 0, 0, 150, 0, 150), contact_cap_rad=0.07)
    assert set(g.contact_ceilings(MEASURED, {**MEASURED, "index_1": 1.2}, 1.0)) == {"index_1"}  # not thumb_1
    assert "index_1" in g.contact_ceilings(MEASURED, {**MEASURED, "index_1": 0.77}, 1.1)  # a small relax
    g.on_force(NAMES, (0, 0, 0, 0, 0, 0), 1.2)
    assert g.contact_ceilings(MEASURED, {**MEASURED, "index_1": 0.7}, 1.2) == {}  # opened 0.1 below
    assert guard_with(force=(0, 0, 0, 150, 0, 0)).contact_ceilings(MEASURED, None, 1.0) == {}  # default off


def test_a_drifting_tip_does_not_pin_a_cap_on_a_free_finger():
    """10.11: the ring tip read 0.2-0.3 N with the hand open after a grasp."""
    g = guard_with(contact_cap_rad=0.07)
    g.on_tips({"ring": 0.3}, 1.0)
    assert g.contact_ceilings(MEASURED, {**MEASURED, "ring_1": 1.2}, 1.0) == {}
    g.on_tips({"ring": 0.8}, 1.1)
    assert set(g.contact_ceilings(MEASURED, {**MEASURED, "ring_1": 1.2}, 1.1)) == {"ring_1"}


def test_no_touch_is_taken_while_the_operator_opens():
    g = guard_with(force=(0, 0, 0, 150, 0, 0), contact_cap_rad=0.07)
    assert g.contact_ceilings(MEASURED, {**MEASURED, "index_1": 0.5}, 1.0) == {}


def test_controller_contact_cap_stops_the_finger_soon_after_the_touch():
    """The glove asks for a fist; the index touches (150 g, under the 950 g hold): it is commanded no
    further than cap past where it touched, instead of creeping on to the fist."""
    rig = Rig(make_controller())
    cfg = rig.ctl.grip_guard.config
    assert cfg.contact_cap_rad > 0  # the station config turns it on
    _, outs = follow_fist_on_a_cup(rig, 150)
    last = outs[-1]
    touch = last.record["contact_touch_rad"]["index_1"]
    assert last.record["q_command_rad"]["index_1"] <= touch + cfg.contact_cap_rad + 1e-3  # touch rounded in the record
    assert last.record["q_command_rad"]["middle_1"] > touch + cfg.contact_cap_rad  # not touching: fist
    rig.ctl.command("home")
    out = rig.step(POSE_ANGLES["fist"])
    assert out.record["contact_touch_rad"] is None


# -- stalled joints (10.11 18:03 arm4090: free fingers read 100-350 g while closing fast and stopped at
#    a cap in free air, 103 "touches", none blocked) --

def stall_guard(**cfg) -> GripGuard:
    return guard_with(force=(0, 0, 0, 200, 0, 0), contact_cap_rad=0.07, contact_stall_rad=0.03,
                      contact_stall_s=0.1, **cfg)


def feed(g, index_path, command, t0=1.0, dt=1 / 120):
    t = t0
    for q in index_path:
        g.observe({**MEASURED, "index_1": q}, {**MEASURED, "index_1": command}, t)
        g.on_force(NAMES, (0, 0, 0, 200, 0, 0), t)
        t += dt
    return t - dt


def test_a_moving_finger_reading_force_is_no_contact():
    g = stall_guard()
    t = feed(g, [0.2 + 0.015 * k for k in range(20)], command=0.6)  # closing at 1.8 rad/s
    assert g.contact_ceilings({**MEASURED, "index_1": 0.485}, {**MEASURED, "index_1": 0.9}, t) == {}
    assert g.speeds(t, 2000)["index_1"] == 2000


def test_a_finger_resting_at_its_command_reading_force_is_no_contact():
    """Bent past 1.1 rad a free finger reads ~110 g at rest: at its command, nothing holds it back."""
    g = stall_guard()
    t = feed(g, [1.25] * 20, command=1.26)
    assert g.contact_ceilings({**MEASURED, "index_1": 1.25}, {**MEASURED, "index_1": 1.26}, t) == {}


def test_a_stalled_finger_behind_its_command_touches():
    g = stall_guard()
    t = feed(g, [0.5 + 0.001 * k for k in range(20)], command=0.7)
    assert g.contact_ceilings({**MEASURED, "index_1": 0.52}, {**MEASURED, "index_1": 0.9}, t) == {
        "index_1": pytest.approx(0.59)}
    assert g.speeds(t, 2000)["index_1"] == 300


def test_a_stall_needs_a_full_window_and_a_tip_needs_none():
    g = stall_guard()
    t = feed(g, [0.5] * 5, command=0.7)  # 0.03 s of history
    assert g.contact_ceilings({**MEASURED, "index_1": 0.5}, {**MEASURED, "index_1": 0.9}, t) == {}
    g.on_tips({"index": 0.6}, t)  # a fingertip press counts at once, moving or not
    assert g.contact_ceilings({**MEASURED, "index_1": 0.5}, {**MEASURED, "index_1": 0.9}, t) == {
        "index_1": pytest.approx(0.57)}


def test_stall_config_is_checked():
    with pytest.raises(ValueError):
        GripGuardConfig(contact_stall_s=0.0)
    assert grip_guard_config({"contact_stall_rad": 0.03}).contact_stall_rad == 0.03


def test_controller_free_fast_close_with_force_does_not_cap():
    """The rig's finger follows its command; a force reading while it closes is no touch."""
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["state"] == "running")
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["phase"] == "follow")
    for _ in range(60):
        press_index(rig, 250)
        out = rig.step(POSE_ANGLES["fist"])
    assert out.record["contact_touch_rad"] is None
    assert out.record["q_command_rad"]["index_1"] > 1.4
