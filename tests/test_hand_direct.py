"""Direct map (10.11 user: "switch to the other project", Nova2Dex): each joint from one SDK value."""

from __future__ import annotations

import pytest

from hand_fixtures import CONFIGURED, make_calibration
from test_hand_controller import HAND_MAP, Rig

from motion_acq.hand.controller import HandController, ControllerConfig
from motion_acq.hand.direct import SDK_PREFIX, DirectConfig, DirectError, DirectRetargeter, direct_config
from motion_acq.hand.retarget import HandRetargeter
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_sdk

LIMITS = CONFIGURED.limits_rad


def sdk(thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0, yaw=0.0) -> dict[str, float]:
    values = {"nflex_thumb": thumb, "nflex_index": index, "nflex_middle": middle, "nflex_ring": ring,
              "nflex_pinky": pinky, "thumb_0_x": yaw, "raw_thumb_abd": 12.0}
    return {SDK_PREFIX + k: v for k, v in values.items()}


NOVA2DEX = DirectConfig(thumb_1_rad=(0.8, 2.0))  # Nova2Dex's numbers, thumb_1 from spread (0.8) to across


def test_configured_method_is_direct_with_the_measured_numbers():
    assert CONFIGURED.method == "direct"
    c = CONFIGURED.direct
    assert c.finger_open["index"] == 0.27 and c.finger_full["ring"] == 0.90 and c.thumb_full == 0.70
    assert c.thumb_yaw_rad == {"right": (0.22, 0.12, -0.17), "left": (-0.34, -0.07, 0.35)}
    assert c.thumb_1_rad == (0.6, 1.57, 2.0943951)  # the robot's whole thumb_1 range, home in the middle
    assert direct_config({"finger_full": 0.9}).finger_full == {f: 0.9 for f in ("index", "middle", "ring", "pinky")}
    with pytest.raises(DirectError, match="unknown keys"):
        direct_config({"finger_range": 1.0})
    with pytest.raises(DirectError):
        direct_config({"thumb_yaw_rad": {"right": [0.1, 0.1], "left": [0.0, 0.3]}})
    with pytest.raises(DirectError):
        direct_config({"finger_open": 0.9, "finger_full": 0.9})


def test_open_hand_full_bend_and_halfway():
    d = DirectRetargeter(NOVA2DEX, LIMITS, "right")
    q = d.solve(sdk())
    assert q["thumb_1"] == pytest.approx(0.8) and q["thumb_2"] == 0.0
    assert all(q[j] == 0.0 for j in ("index_1", "middle_1", "ring_1", "pinky_1"))
    q = d.solve(sdk(0.733333, 0.928571, 0.928571, 0.928571, 0.928571, -0.3))
    assert q["thumb_1"] == pytest.approx(2.0)
    assert q["thumb_2"] == pytest.approx(LIMITS["thumb_2"][1])
    assert q["index_1"] == pytest.approx(LIMITS["index_1"][1]) and q["pinky_1"] == pytest.approx(LIMITS["pinky_1"][1])
    q = d.solve(sdk(index=0.928571 / 2, yaw=-0.15))
    assert q["index_1"] == pytest.approx(LIMITS["index_1"][1] / 2)
    assert q["thumb_1"] == pytest.approx(1.4)
    assert q["middle_1"] == 0.0  # one finger drives one joint: no crosstalk term at all


def test_values_past_the_ends_clip():
    d = DirectRetargeter(NOVA2DEX, LIMITS, "right")
    q = d.solve(sdk(thumb=1.0, index=1.0, yaw=-0.9))
    assert q["index_1"] == pytest.approx(LIMITS["index_1"][1]) and q["thumb_1"] == pytest.approx(2.0)
    q = d.solve(sdk(yaw=0.4))
    assert q["thumb_1"] == pytest.approx(0.8)


def test_measured_open_hand_is_straight_and_fist_is_closed():
    """10.11 recording medians: the open hand reads 0.14-0.33, not 0, and must still send the hand straight."""
    for side, open_, fist, across in (("right", (0.30, 0.27, 0.33, 0.17, 0.12), (0.60, 0.84, 0.88, 0.90, -0.07),
                                       (0.59, 0.18, 0.31, 0.15, -0.08)),
                                      ("left", (0.20, 0.23, 0.28, 0.14, -0.07), (0.65, 0.80, 0.88, 0.90, 0.22),
                                       (0.55, 0.17, 0.21, 0.09, 0.19))):
        d = DirectRetargeter(CONFIGURED.direct, LIMITS, side)
        t, i, m, r, x = open_
        q = d.solve(sdk(t, i, m, r, r, x))
        assert q["thumb_1"] == pytest.approx(1.57) and max(q[j] for j in ("thumb_2", "index_1", "middle_1", "ring_1")) < 0.01
        t, i, m, r, x = fist
        q = d.solve(sdk(t, i, m, r, r, x))
        assert min(q[j] for j in ("index_1", "middle_1", "ring_1", "pinky_1")) > 1.4
        t, i, m, r, x = across
        q = d.solve(sdk(t, i, m, r, r, x))
        assert 1.8 < q["thumb_1"] < 2.0 and q["index_1"] < 0.05  # the 20:42 across pose, short of today's ends


def test_each_glove_has_its_own_thumb_rotation_ends():
    right = DirectRetargeter(CONFIGURED.direct, LIMITS, "right")
    left = DirectRetargeter(CONFIGURED.direct, LIMITS, "left")
    # 10.11 user: thumb_1's whole range; beside (the 20:42 pose) stays home
    for d, (spread, beside, across) in ((right, (0.22, 0.12, -0.17)), (left, (-0.34, -0.07, 0.35))):
        assert d.solve(sdk(yaw=beside))["thumb_1"] == pytest.approx(1.57)
        assert d.solve(sdk(yaw=across))["thumb_1"] == pytest.approx(2.0943951)
        assert d.solve(sdk(yaw=spread))["thumb_1"] == pytest.approx(0.6)
        assert d.solve(sdk(yaw=across + (across - beside)))["thumb_1"] == pytest.approx(2.0943951)  # clipped
        half = d.solve(sdk(yaw=(beside + across) / 2))["thumb_1"]
        assert half == pytest.approx((1.57 + 2.0943951) / 2)
    assert left.solve(sdk(yaw=-0.2))["thumb_1"] < 1.57 < left.solve(sdk(yaw=0.0))["thumb_1"]


def test_missing_sdk_values_say_which_node_to_start():
    d = DirectRetargeter(DirectConfig(), LIMITS, "right")
    signals = sdk()
    del signals[SDK_PREFIX + "thumb_0_x"]
    assert not d.ready(signals)
    with pytest.raises(DirectError, match="nova2_sdk"):
        d.solve(signals)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("pose", ["flat", "fist", "index", "thumb_bend", "thumb_opposed"])
def test_fake_glove_sdk_values_map_back_to_its_robot_pose(side, pose):
    angles = POSE_ANGLES[pose]
    for config in (NOVA2DEX, CONFIGURED.direct):
        q = DirectRetargeter(config, LIMITS, side).solve(
            {SDK_PREFIX + k: v for k, v in synthetic_sdk(angles, side, config).items()})
        for j in ("thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"):
            assert q[j] == pytest.approx(min(angles[f"q_{j}"], LIMITS[j][1]), abs=1e-6)  # synthetic ends 1e-4 past
        lo, hi = config.thumb_1_rad[0], config.thumb_1_rad[-1]
        assert q["thumb_1"] == pytest.approx(min(max(angles["q_thumb_1"], lo), hi), abs=1e-6)


def make_direct(side: str = "right") -> HandController:
    rt = HandRetargeter(CONFIGURED, make_calibration(side), HAND_MAP, side)
    return HandController(rt, HAND_MAP, side, ControllerConfig())


def test_controller_follows_the_sdk_without_a_reference_take():
    """Enable walks home and then follows at once: the SDK calibrates the glove, nothing of ours to re-zero."""
    rig = Rig(make_direct())
    rig.ctl.request_enable(True)
    phases = []
    for _ in range(150):
        out = rig.step(sdk(0.30, 0.27, 0.33, 0.17, 0.17, 0.12))  # the measured open hand
        phases.append(out.record["phase"])
    assert "reference" not in phases and phases[-1] == "follow"
    for _ in range(150):
        out = rig.step(sdk(thumb=0.70, index=0.9, middle=0.9, ring=0.9, pinky=0.9, yaw=-0.08))
    assert out.record["method"] == "direct"
    q = out.record["q_command_rad"]
    assert q["index_1"] > 1.4 and q["ring_1"] > 1.4 and q["thumb_2"] > 0.4 and q["thumb_1"] > 1.8
    assert out.record["features"]["unit_index_1"] == pytest.approx(1.0)


def test_controller_holds_when_the_sdk_stops():
    rig = Rig(make_direct())
    rig.ctl.request_enable(True)
    for _ in range(150):
        rig.step(sdk(index=0.5))
    for _ in range(30):
        out = rig.step(None)
    assert out.record["state"] == "hold"
