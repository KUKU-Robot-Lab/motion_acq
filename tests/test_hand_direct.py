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


def test_configured_method_is_direct_with_nova2dex_numbers():
    assert CONFIGURED.method == "direct"
    c = CONFIGURED.direct
    assert (c.finger_full, c.thumb_full, c.thumb_yaw_rad) == (0.928571, 0.733333, (0.0, -0.3))
    with pytest.raises(DirectError, match="unknown keys"):
        direct_config({"finger_range": 1.0})
    with pytest.raises(DirectError):
        DirectConfig(thumb_yaw_rad=(0.1, 0.1))


def test_open_hand_full_bend_and_halfway():
    d = DirectRetargeter(DirectConfig(), LIMITS, "right")
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
    d = DirectRetargeter(DirectConfig(), LIMITS, "right")
    q = d.solve(sdk(thumb=1.0, index=1.0, yaw=-0.9))
    assert q["index_1"] == pytest.approx(LIMITS["index_1"][1]) and q["thumb_1"] == pytest.approx(2.0)
    q = d.solve(sdk(yaw=0.4))
    assert q["thumb_1"] == pytest.approx(0.8)


def test_left_glove_reads_the_thumb_rotation_mirrored():
    right = DirectRetargeter(DirectConfig(), LIMITS, "right")
    left = DirectRetargeter(DirectConfig(), LIMITS, "left")
    assert left.solve(sdk(yaw=0.3))["thumb_1"] == pytest.approx(right.solve(sdk(yaw=-0.3))["thumb_1"])
    assert left.solve(sdk(yaw=-0.3))["thumb_1"] == pytest.approx(0.8)


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
    q = DirectRetargeter(DirectConfig(), LIMITS, side).solve(
        {SDK_PREFIX + k: v for k, v in synthetic_sdk(angles, side).items()})
    for j in ("thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"):
        assert q[j] == pytest.approx(min(angles[f"q_{j}"], LIMITS[j][1]), abs=1e-6)  # synthetic ends are 1e-4 past
    assert q["thumb_1"] == pytest.approx(min(max(angles["q_thumb_1"], 0.8), 2.0), abs=1e-6)


def make_direct(side: str = "right") -> HandController:
    rt = HandRetargeter(CONFIGURED, make_calibration(side), HAND_MAP, side)
    return HandController(rt, HAND_MAP, side, ControllerConfig())


def test_controller_follows_the_sdk_without_a_reference_take():
    """Enable walks home and then follows at once: the SDK calibrates the glove, nothing of ours to re-zero."""
    rig = Rig(make_direct())
    rig.ctl.request_enable(True)
    phases = []
    for _ in range(150):
        out = rig.step(sdk())
        phases.append(out.record["phase"])
    assert "reference" not in phases and phases[-1] == "follow"
    for _ in range(150):
        out = rig.step(sdk(thumb=0.733333, index=0.928571, middle=0.928571, ring=0.928571, pinky=0.928571, yaw=-0.3))
    assert out.record["method"] == "direct"
    q = out.record["q_command_rad"]
    assert q["index_1"] > 1.4 and q["ring_1"] > 1.4 and q["thumb_2"] > 0.4 and q["thumb_1"] > 1.9
    assert out.record["features"]["unit_index_1"] == pytest.approx(1.0)


def test_controller_holds_when_the_sdk_stops():
    rig = Rig(make_direct())
    rig.ctl.request_enable(True)
    for _ in range(150):
        rig.step(sdk(index=0.5))
    for _ in range(30):
        out = rig.step(None)
    assert out.record["state"] == "hold"
