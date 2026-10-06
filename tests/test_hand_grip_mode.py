"""Position + force closed loop per finger (10.06, docs/RH56F1_HAND_TUNING.md)."""

from __future__ import annotations

import pytest
from test_hand_controller import Rig, make_controller, run_until

from motion_acq.hand.grip_mode import GripMode, GripModeConfig, GripState, grip_mode_config
from motion_acq.hand.synthetic import POSE_ANGLES

FREE_MODES = {j: 0 for j in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")}


def gm(**cfg) -> GripMode:
    g = GripMode(GripModeConfig(**cfg))
    g.on_modes(FREE_MODES)
    return g


def step(g, t, cmd=1.0, meas=0.8, force=300.0, tip=0.0, current=60.0, joint="index_1"):
    return g.update(t, {joint: cmd}, {joint: meas}, {joint: force}, {joint: current}, {"index": tip, "thumb": tip})


def test_contact_and_operator_closing_past_switches_to_force():
    g = gm()
    out = step(g, 0.0)
    assert out.modes == {"index_1": 1} and g.state["index_1"] is GripState.ENTERING
    assert "index_1" in out.pinned and 150 <= out.force_g["index_1"] <= 800
    g.on_modes({**FREE_MODES, "index_1": 1})
    out = step(g, 0.01)
    assert g.state["index_1"] is GripState.HOLD and out.modes is None


def test_no_switch_without_contact_or_without_closing_past():
    g = gm()
    assert step(g, 0.0, force=50.0).modes is None
    assert step(g, 0.0, cmd=0.82).modes is None  # touching but the operator does not close further


def test_no_switch_with_a_driver_that_does_not_report_modes():
    g = GripMode(GripModeConfig())
    assert step(g, 0.0).modes is None and not g.active()


def test_grip_force_grows_with_how_far_the_operator_closes_and_is_capped():
    g = gm()
    a = step(g, 0.0, cmd=0.8501, tip=1.0).force_g["index_1"]
    g2 = gm()
    b = step(g2, 0.0, cmd=1.0, tip=1.0).force_g["index_1"]
    g3 = gm()
    c = step(g3, 0.0, cmd=3.0, tip=1.0).force_g["index_1"]
    assert a == pytest.approx(300, abs=1) and a < b < c == pytest.approx(800)


def test_link_1_contact_and_thumb_scale():
    g = gm()
    tip = step(g, 0.0, tip=1.0).force_g["index_1"]
    g2 = gm()
    link1 = step(g2, 0.0, tip=0.0).force_g["index_1"]
    assert link1 == pytest.approx(0.7 * tip)
    g3 = gm()
    thumb = step(g3, 0.0, tip=1.0, joint="thumb_2").force_g["thumb_2"]
    assert thumb == pytest.approx(1.55 * tip)


def test_operator_opening_returns_to_position_after_confirmation():
    g = gm()
    step(g, 0.0)
    g.on_modes({**FREE_MODES, "index_1": 1})
    step(g, 0.01)
    out = step(g, 0.1, cmd=0.70)  # opened 0.1 rad short, but held only 0.09 s: stay
    assert out.modes is None
    out = step(g, 0.3, cmd=0.70)
    assert out.modes == {"index_1": 0} and g.state["index_1"] is GripState.LEAVING and "index_1" in out.pinned
    g.on_modes(FREE_MODES)
    out = step(g, 0.31, cmd=0.70)
    assert g.state["index_1"] is GripState.FREE and "index_1" not in out.pinned


def test_unconfirmed_switch_is_undone():
    g = gm(ack_timeout_s=0.5)
    step(g, 0.0)
    out = step(g, 0.6)
    assert out.modes == {"index_1": 0} and g.failures == 1


def test_release_all_when_leaving_follow():
    g = gm()
    step(g, 0.0)
    g.on_modes({**FREE_MODES, "index_1": 1})
    step(g, 0.01)
    out = g.release_all(0.02)
    assert out.modes == {"index_1": 0} and out.pinned == {"index_1"}
    g.on_modes(FREE_MODES)
    assert g.release_all(0.03).pinned == set() and not g.active()


def test_config_checks():
    with pytest.raises(ValueError, match="cannot hold"):
        grip_mode_config({"joints": ["thumb_1"]})
    with pytest.raises(ValueError):
        grip_mode_config({"max_g": 2000})


# -- in the controller ------------------------------------------------------------------------------

NAMES = ["r_hj_pinky_1", "r_hj_ring_1", "r_hj_middle_1", "r_hj_index_1", "r_hj_thumb_2", "r_hj_thumb_1"]


class ModeHand(Rig):
    """Rig whose fake firmware takes mode requests and holds the index on a cup."""

    def __init__(self, ctl):
        super().__init__(ctl)
        self.modes = [0] * 6
        self.cup = None

    def step(self, angles=None, **kw):
        k = len(self.sent) % 2
        idx_force = 500 if self.cup is not None and self.hand[3] <= self.cup + 2 else 10
        self.ctl.on_joint_force(NAMES, [10, 10, 10, idx_force + k, 10, 10], self.t)
        self.ctl.on_current(NAMES, [60, 60, 60, 60 + k, 60, 60], self.t)
        self.ctl.on_finger_mode(self.modes)
        out = super().step(angles, **kw)
        if out.mode is not None:
            self.modes = [m if m >= 0 else old for m, old in zip(out.mode, self.modes)]
        if self.cup is not None and self.modes[3] == 0:
            self.hand[3] = max(self.hand[3], self.cup)
        if self.modes[3] == 1 and self.cup is not None:
            self.hand[3] = self.cup  # force loop holds it on the cup
        return out


def test_controller_holds_by_force_and_releases_to_position():
    rig = ModeHand(make_controller())
    rig.ctl.request_enable(True)
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["state"] == "running")
    out = run_until(rig, POSE_ANGLES["fist"], lambda o: o.angle[3] < 1450)
    rig.cup = out.angle[3]
    out = run_until(rig, POSE_ANGLES["fist"], lambda o: rig.modes[3] == 1)
    for _ in range(10):
        out = rig.step(POSE_ANGLES["fist"])
    assert out.record["grip_mode"]["state"] == {"index_1": "hold"}
    assert out.angle[3] == rig.hand[3]  # pinned at the measured angle
    assert out.force is None or out.force[3] <= 1000
    assert out.angle[2] < 1000  # the middle finger still follows the fist
    for _ in range(40):
        out = rig.step(POSE_ANGLES["flat"])
    assert rig.modes[3] == 0 and out.record["grip_mode"] is None
    assert out.angle[3] > rig.cup + 100  # opened with the glove


def test_disable_returns_held_fingers_to_position_first():
    rig = ModeHand(make_controller())
    rig.ctl.request_enable(True)
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["state"] == "running")
    out = run_until(rig, POSE_ANGLES["fist"], lambda o: o.angle[3] < 1450)
    rig.cup = out.angle[3]
    run_until(rig, POSE_ANGLES["fist"], lambda o: rig.modes[3] == 1)
    rig.ctl.request_enable(False)
    out = rig.step(POSE_ANGLES["fist"])
    assert rig.modes[3] == 0
