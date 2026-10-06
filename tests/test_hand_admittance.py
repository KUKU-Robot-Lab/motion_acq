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
    a, b = Admittance(), Admittance()
    tip, _ = run(a, 440.0, 2.0, tip=1.0)
    link1, _ = run(b, 440.0, 2.0, tip=0.0)
    assert link1["index_1"] == pytest.approx(tip["index_1"] / 0.7, rel=0.01)


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
