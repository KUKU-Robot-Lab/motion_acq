"""Position-based admittance per finger (10.06 user: position + force loop together, not switched)."""

from __future__ import annotations

import pytest
from test_hand_controller import Rig, run_until

from motion_acq.hand.admittance import Admittance, AdmittanceConfig, admittance_config
from motion_acq.hand.synthetic import POSE_ANGLES


def run(adm: Admittance, force: float, seconds: float, tip: float = 1.0, joint: str = "index_1", t0: float = 0.0):
    y, t = {}, t0
    for _ in range(int(seconds * 120)):
        t += 1 / 120
        y = adm.update(t, {joint: force}, {"index": tip, "thumb": tip})
    return y, t


def test_free_space_follows_the_operator():
    adm = Admittance()
    y, _ = run(adm, 20.0, 1.0)  # below the deadband
    assert y == {}


def test_offset_settles_at_force_over_stiffness():
    adm = Admittance(AdmittanceConfig(stiffness_g_per_rad=2000.0, deadband_g=40.0, filter_tau_s=0.15))
    y, _ = run(adm, 440.0, 2.0)
    assert y["index_1"] == pytest.approx(400.0 / 2000.0, rel=0.01)


def test_link_1_contact_counts_more_force():
    cfg = AdmittanceConfig(proximal_scale=0.7)
    a, b = Admittance(cfg), Admittance(cfg)
    tip, _ = run(a, 440.0, 2.0, tip=1.0)
    link1, _ = run(b, 440.0, 2.0, tip=0.0)
    assert link1["index_1"] == pytest.approx(tip["index_1"] / 0.7, rel=0.01)


def test_default_reading_is_the_grip_force_wherever_it_touches():
    """10.06 user (A): no link 1 boost by default."""
    a, b = Admittance(), Admittance()
    tip, _ = run(a, 440.0, 2.0, tip=1.0)
    link1, _ = run(b, 440.0, 2.0, tip=0.0)
    assert link1["index_1"] == pytest.approx(tip["index_1"], rel=1e-6)


def test_filter_and_cap():
    adm = Admittance(AdmittanceConfig(filter_tau_s=0.15, max_offset_rad=0.3))
    y, t = run(adm, 2000.0, 2 / 120)
    assert 0 < y["index_1"] < 0.1  # one tick: only part of the way
    y, _ = run(adm, 5000.0, 3.0, t0=t)
    assert y["index_1"] == pytest.approx(0.3, rel=0.01)


def test_release_when_the_force_goes():
    adm = Admittance()
    _, t = run(adm, 640.0, 2.0)
    y, _ = run(adm, 0.0, 2.0, t0=t)
    assert y == {}


def test_stiff_contact_converges_without_oscillation():
    """Hand + object as a spring: force = k * max(q_cmd - q_contact, 0), k up to ~29 kg/rad (thumb_2)."""
    for k in (1600.0, 8000.0, 29000.0):
        adm = Admittance()
        q_op, q_c, q_cmd, t, forces = 1.0, 0.7, 1.0, 0.0, []
        for _ in range(600):
            t += 1 / 120
            f = k * max(q_cmd - q_c, 0.0)
            y = adm.update(t, {"index_1": f}, {"index": 1.0}).get("index_1", 0.0)
            q_cmd = q_op - y
            forces.append(f)
        tail = forces[-120:]
        assert max(tail) - min(tail) < 5.0, k  # settled
        assert 200 < tail[-1] < 900, k  # stiffness in series with the contact x 0.3 rad, never the saturation


def test_config_checks():
    with pytest.raises(ValueError, match="force sign"):
        admittance_config({"joints": ["thumb_1"]})
    with pytest.raises(ValueError):
        admittance_config({"stiffness_g_per_rad": 0})


NAMES = ["r_hj_pinky_1", "r_hj_ring_1", "r_hj_middle_1", "r_hj_index_1", "r_hj_thumb_2", "r_hj_thumb_1"]


def test_controller_grip_force_follows_how_far_the_operator_closes():
    """motion_acq's own 120 Hz admittance (driver.command angle_set; the default is the driver's, angle_target).
    A spring cup at the index: the held force ends near stiffness x penetration, not at saturation."""
    import dataclasses

    import test_hand_controller as thc
    from hand_fixtures import CONFIG, make_calibration

    from motion_acq.hand.controller import ControllerConfig, HandController
    from motion_acq.hand.retarget import HandRetargeter

    cfg = dataclasses.replace(CONFIG, reference_s=0.0, driver_command="angle_set",
                              admittance=dataclasses.replace(CONFIG.admittance, enabled=True))
    rt = HandRetargeter(cfg, make_calibration("right"), thc.HAND_MAP, "right")
    rig = Rig(HandController(rt, thc.HAND_MAP, "right", ControllerConfig()))
    rig.ctl.request_enable(True)
    run_until(rig, POSE_ANGLES["flat"], lambda o: o.record["state"] == "running")
    out = run_until(rig, POSE_ANGLES["fist"], lambda o: o.angle[3] < 1450)
    cup = out.angle[3] + 40
    k_reg = 8.0  # g per register past the cup
    for n in range(240):
        f = max(cup - rig.hand[3], 0) * k_reg
        rig.ctl.on_joint_force(NAMES, [10, 10, 10, f + n % 2, 10, 10], rig.t)
        rig.ctl.on_current(NAMES, [60, 60, 60, 60 + n % 2, 60, 60], rig.t)
        rig.ctl.on_touch([0, 0, 0, 200, 0], [0] * 9, rig.t)
        out = rig.step(POSE_ANGLES["fist"])
    force = max(cup - rig.hand[3], 0) * k_reg
    assert out.record["admittance"]["index_1"]["offset_rad"] > 0.1
    assert 100 < force < 950  # held near the admittance ceiling, below the guard, far from 1.1-1.85 kg
    assert out.angle[2] < 1000  # free fingers still reach the fist


def test_driver_admittance_is_the_default_and_excludes_ours():
    from hand_fixtures import CONFIG

    assert CONFIG.driver_command == "angle_target" and not CONFIG.admittance.enabled


def test_stiff_contact_skips_the_soft_first_touch():
    from motion_acq.hand.admittance import stiff_contact

    reg = 1 / 550
    soft = [(0.6 + i * reg, 120.0 + 6.0 * i) for i in range(40)]           # cup giving way, 6 g/register
    q_stiff = soft[-1][0]
    rigid = [(q_stiff + i * reg, soft[-1][1] + 55.0 * i) for i in range(1, 5)]  # 55 g/register
    assert q_stiff - 0.01 <= stiff_contact(soft + rigid) <= q_stiff  # early by <= the span (~20 g at 2000 g/rad)
    assert stiff_contact(soft) is None
    stall = [(0.6, 120.0), (0.6 + reg, 130.0), (0.6 + 3 * reg, 150.0), (0.6 + 3 * reg, 300.0)]
    assert stiff_contact(stall) == pytest.approx(0.6, abs=1e-9)  # the trace ends in a stall: slope to the last sample


def test_contact_ceiling_restarts_from_the_finger_and_closes_at_the_rate():
    """10.06: a lead cap held a rigid cup at ~350 g; now the command closes at a bounded rate, no cap."""
    adm = Admittance(AdmittanceConfig(filter_tau_s=1.0))
    adm.update(0.0, {"index_1": 90.0}, {})                       # 50 g over the deadband: fast free-space closing
    assert adm.ceilings(0.0, {"index_1": 0.6}) == {}
    adm.update(0.01, {"index_1": 440.0}, {})                     # 400 g: contact
    assert adm.ceilings(0.01, {"index_1": 0.6}) == {"index_1": pytest.approx(0.6 + 0.3 * 0.5 * 0.01)}
    top = adm.ceilings(0.02, {"index_1": 0.6})["index_1"]
    assert top == pytest.approx(0.6 + 2 * 0.3 * 0.5 * 0.01)      # keeps closing, no cap vs the finger
    assert adm.ceilings(0.03, {"index_1": 0.6}, {"index_1": 0.4})["index_1"] == pytest.approx(0.4 + 0.0015)  # opened
    for k in range(1, 41):                                       # force gone, offset fades (0.15 s)
        adm.update(0.03 + 0.05 * k, {"index_1": 0.0}, {})
        last = adm.ceilings(0.03 + 0.05 * k, {"index_1": 0.6})
    assert last == {}                                            # free again
