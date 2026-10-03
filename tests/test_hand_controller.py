"""STEP 3 hand controller: enable rules, faults, freeze, hand_id, parity with sim2real."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml

from motion_acq.hand.calibration import CalibrationError, HandCalibration, calibrate
from motion_acq.hand.controller import (
    ControllerConfig,
    HandController,
    Mode,
    implausible_registers,
)
from motion_acq.hand.nova2 import features
from motion_acq.hand.retarget import HandRetargeter, load_hand_retarget_config
from motion_acq.hand.rh56f1 import load_rh56f1_map
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles

CONFIG = load_hand_retarget_config()
HAND_MAP = load_rh56f1_map()
DT = 1 / 30


def make_calibration(side: str) -> HandCalibration:
    samples = {pose: [features(a, CONFIG.features)] for pose, a in POSE_ANGLES.items()}
    return calibrate(side=side, user="t", pose_samples=samples,
                     feature_poses=CONFIG.feature_poses, min_span=CONFIG.min_span_rad)


def make_controller(side: str = "right", amplitude: float = 1.0, **cfg) -> HandController:
    rt = HandRetargeter(CONFIG, make_calibration(side), HAND_MAP, side, amplitude=amplitude)
    return HandController(rt, HAND_MAP, side, ControllerConfig(**cfg))


HOME_R = HAND_MAP.to_registers(CONFIG.home_rad, side="right")


class Rig:
    """Drives a controller with a glove and a hand that follows its commands."""

    def __init__(self, ctl: HandController, hand_id: int = 3) -> None:
        self.ctl, self.t, self.hand_id = ctl, 0.0, hand_id
        self.hand = list(HOME_R)
        self.sent: list = []

    def step(self, angles=None, *, measured=True, subscribers=True, jitter=True):
        self.t += DT
        if angles is not None:
            if jitter:  # a worn glove never repeats exactly (frozen data means SenseCom died)
                noise = 1e-5 if len(self.sent) % 2 else -1e-5
                angles = {k: v + noise for k, v in angles.items()}
            self.ctl.on_glove(angles, self.t)
        if measured:
            self.ctl.on_measured(self.hand, self.hand_id, self.t)
        out = self.ctl.tick(self.t, subscribers_ready=subscribers)
        if out.angle is not None:
            self.hand = [v if v != -1 else h for v, h in zip(out.angle, self.hand, strict=True)]
        self.sent.append(out)
        return out


def test_disabled_publishes_nothing():
    rig = Rig(make_controller())
    for _ in range(10):
        out = rig.step(POSE_ANGLES["fist"])
        assert out.angle is None and out.speed is None
    assert rig.ctl.mode is Mode.DISABLED


@pytest.mark.parametrize(
    ("measured", "subscribers", "reason"),
    [(False, True, "no fresh angle_actual"), (True, False, "no subscriber")],
)
def test_enable_needs_feedback_and_subscribers(measured, subscribers, reason):
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    out = rig.step(POSE_ANGLES["open"], measured=measured, subscribers=subscribers)
    assert rig.ctl.mode is Mode.DISABLED and out.angle is None
    assert reason in out.record["refusal"]


@pytest.mark.parametrize("bad", [[0] * 6, [-1] * 6, [65535] * 6, [1740, 1740, 1740, 1740, 1350, 300]])
def test_enable_refuses_implausible_feedback(bad):
    rig = Rig(make_controller())
    rig.hand = list(bad)
    rig.ctl.request_enable(True)
    out = rig.step(POSE_ANGLES["open"])
    assert rig.ctl.mode is Mode.DISABLED and out.angle is None
    assert "implausible" in out.record["refusal"]
    assert implausible_registers(bad, HAND_MAP, "right")


def test_enable_starts_at_measured_pose_with_driver_hand_id():
    rig = Rig(make_controller())
    rig.hand = [1300, 1250, 1200, 1150, 1250, 1200]
    rig.ctl.request_enable(True)
    out = rig.step(POSE_ANGLES["fist"])
    assert rig.ctl.mode is Mode.ENABLED
    assert out.hand_id == 3
    assert out.speed == [2000] * 6 and out.force == [600] * 6
    assert max(abs(a - b) for a, b in zip(out.angle, [1300, 1250, 1200, 1150, 1250, 1200], strict=True)) <= 2


def test_speed_and_force_are_resent_every_second():
    rig = Rig(make_controller(driver_speed=1000, driver_force=300))
    rig.ctl.request_enable(True)
    for _ in range(95):
        rig.step(POSE_ANGLES["open"])
    resends = [i for i, o in enumerate(rig.sent) if o.speed is not None]
    assert len(resends) == 4  # t = 0, 1, 2, 3 s (95 ticks at 30 Hz)
    assert rig.sent[resends[0]].speed == [1000] * 6 and rig.sent[resends[0]].force == [300] * 6


def test_driver_limits_are_validated():
    with pytest.raises(ValueError):
        ControllerConfig(driver_speed=5000)
    with pytest.raises(ValueError):
        ControllerConfig(driver_force=0)


def test_glove_hold_publishes_nothing_and_resumes_without_jump():
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    for _ in range(60):
        rig.step(synthetic_angles(0.5, 0.5, 0.5))
    states = []
    for _ in range(30):  # glove silent for 1 s; the last sample stays valid for stale_s
        out = rig.step(None)
        states.append(out.record["state"])
        if out.record["state"] == "hold":
            assert out.angle is None
    assert states.count("hold") >= 30 - math.ceil(0.2 / DT) - 1
    last = next(o.angle for o in reversed(rig.sent) if o.angle is not None)
    out = rig.step(synthetic_angles(1.0, 1.0, 1.0))
    # From rest, one cycle moves at most a*dt^2 rad (~1.3 deg ~ 13 registers).
    step_regs = math.ceil(math.degrees(CONFIG.max_acceleration_rad_s2 * DT * DT) * 10 / 0.98) + 1
    assert max(abs(a - b) for a, b in zip(out.angle, last, strict=True)) <= step_regs


def test_feedback_loss_latches_fault_until_reenabled():
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    for _ in range(10):
        rig.step(POSE_ANGLES["open"])
    for _ in range(20):
        out = rig.step(POSE_ANGLES["fist"], measured=False)
    assert rig.ctl.mode is Mode.FAULT and out.angle is None
    assert "angle_actual lost" in out.record["fault"]
    out = rig.step(POSE_ANGLES["fist"])  # feedback back: still latched
    assert rig.ctl.mode is Mode.FAULT and out.angle is None
    rig.ctl.request_enable(False)
    rig.ctl.request_enable(True)
    out = rig.step(POSE_ANGLES["fist"])
    assert rig.ctl.mode is Mode.ENABLED and out.angle is not None


def test_disable_freezes_the_hand_once():
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    for _ in range(20):
        rig.step(POSE_ANGLES["fist"])
    rig.ctl.request_enable(False)
    out = rig.step(POSE_ANGLES["fist"])
    assert out.angle == rig.sent[-2].angle or out.angle == rig.hand  # the measured pose
    assert rig.ctl.mode is Mode.DISABLED
    assert all(rig.step(POSE_ANGLES["fist"]).angle is None for _ in range(10))


def test_reenable_reseeds_from_the_new_measured_pose():
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    for _ in range(30):
        rig.step(POSE_ANGLES["fist"])
    rig.ctl.request_enable(False)
    rig.step(None)
    rig.hand = list(HOME_R)  # someone moved the hand while disabled
    rig.ctl.request_enable(True)
    out = rig.step(POSE_ANGLES["fist"])
    assert max(abs(a - b) for a, b in zip(out.angle, HOME_R, strict=True)) <= 2


def test_amplitude_scales_travel():
    rig = Rig(make_controller(amplitude=0.3))
    rig.ctl.request_enable(True)
    for _ in range(120):
        out = rig.step(POSE_ANGLES["fist"])
    assert out.record["q_command_rad"]["index_1"] == pytest.approx(0.3 * 1.5285594, abs=1e-3)
    with pytest.raises(ValueError):
        make_controller(amplitude=1.5)


@pytest.mark.parametrize("side", ["right", "left"])
def test_thumb_rotation_register_rises_with_opposition(side):
    regs = [HAND_MAP.to_registers({**CONFIG.home_rad, "thumb_1": q}, side=side)[5]
            for q in (1.57, 1.2, 0.8, 0.3)]
    assert regs == sorted(regs) and regs[0] < regs[-1]


def test_left_hand_full_range():
    rig = Rig(make_controller("left"))
    rig.hand = HAND_MAP.to_registers(CONFIG.home_rad, side="left")
    rig.ctl.request_enable(True)
    for _ in range(120):
        out = rig.step(POSE_ANGLES["fist"])
    assert out.angle[:4] == HAND_MAP.to_registers(out.record["q_command_rad"], side="left")[:4]
    assert all(900 <= v <= 960 for v in out.angle[:4])


def test_calibration_file_with_nan_or_zero_span_is_rejected(tmp_path):
    good = make_calibration("right")
    for bad in ({"open": 0.5, "closed": 0.5}, {"open": float("nan"), "closed": 1.0}):
        data = good.to_dict()
        data["ranges"]["index"] = bad
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(CalibrationError, match="not a usable range"):
            HandCalibration.load(path)


def test_calibration_missing_a_feature_is_rejected():
    cal = make_calibration("right")
    partial = HandCalibration("right", "t", {k: v for k, v in cal.ranges.items() if k != "thumb_bend"})
    with pytest.raises(ValueError, match="lacks features"):
        HandRetargeter(CONFIG, partial, HAND_MAP, "right")


SIM2REAL = Path.home() / "rl_ws/sim2real/deploy/policy_control"


@pytest.mark.skipif(not (SIM2REAL / "policy_control/rh56f1_map.py").exists(), reason="sim2real not present")
def test_register_parity_with_sim2real():
    spec = importlib.util.spec_from_file_location("s2r_map", SIM2REAL / "policy_control/rh56f1_map.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s2r_map"] = mod
    spec.loader.exec_module(mod)
    s2r = mod.load(SIM2REAL / "config/rh56f1_hand_map.yaml")
    import random

    rng = random.Random(0)
    for side in (None, "right", "left"):
        for _ in range(1000):
            q = [rng.uniform(-0.2, 2.3) for _ in range(6)]
            ours = HAND_MAP.to_registers(dict(zip(HAND_MAP.joint_order, q, strict=True)), side=side)
            assert ours == s2r.to_register(q, side=side)
            regs = [rng.randint(500, 1900) for _ in range(6)]
            theirs = s2r.to_rad(regs, side=side)
            mine = HAND_MAP.to_rad(regs, side=side)
            assert all(math.isclose(mine[n], float(t)) for n, t in zip(HAND_MAP.joint_order, theirs, strict=True))


def test_frozen_glove_holds_until_it_changes_again():
    """Dead SenseCom: senseglove_ros republishes the last values on time (bumsu 09-22)."""
    rig = Rig(make_controller())
    rig.ctl.request_enable(True)
    frozen = synthetic_angles(0.6, 0.3, 0.2)
    for _ in range(int(0.9 / DT)):  # identical samples for 0.9 s: still RUNNING
        out = rig.step(frozen, jitter=False)
    assert out.record["state"] == "running" and not out.record["glove_frozen"]
    for _ in range(int(0.3 / DT)):  # past glove_frozen_s (1.0 s): HOLD, nothing sent
        out = rig.step(frozen, jitter=False)
    assert out.record["state"] == "hold" and out.record["glove_frozen"] and out.angle is None
    out = rig.step(synthetic_angles(0.61, 0.3, 0.2))
    assert out.record["state"] == "running" and out.angle is not None
