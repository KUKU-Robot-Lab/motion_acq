"""STEP 2 head retargeting: angles, filters, limits, anchoring and HOLD."""

from __future__ import annotations

import numpy as np
import pytest
from head_helpers import make_config, pose_from_yaw_pitch

from motion_acq.head.retarget import (
    AxisConfig,
    Deadband,
    HeadRetargeter,
    HeadState,
    OneEuroFilter,
    RateLimiter,
    hmd_yaw_pitch_deg,
    wrap_deg,
)
from motion_acq.tracking.mock_quest_sender import HmdMotion
from motion_acq.tracking.transforms import unity_quaternion_to_handumi


@pytest.mark.parametrize(("yaw", "pitch"), [(0, 0), (25, 0), (-40, 10), (10, -30), (170, 5)])
def test_yaw_pitch_roundtrip(yaw, pitch):
    angles = hmd_yaw_pitch_deg(pose_from_yaw_pitch(yaw, pitch))
    assert angles == pytest.approx((yaw, pitch), abs=1e-6)


def test_mock_unity_motion_signs_match_tracking_frame():
    # Unity yaw right / pitch down must read as yaw < 0 / pitch < 0.
    rot = HmdMotion(yaw_amp_deg=30.0, pitch_amp_deg=0.0, period_s=4.0).rotation(1.0)
    q = unity_quaternion_to_handumi([rot["x"], rot["y"], rot["z"], rot["w"]])
    yaw, pitch = hmd_yaw_pitch_deg(np.concatenate([[0, 0, 0], q]))
    assert (yaw, pitch) == pytest.approx((-30.0, 0.0), abs=1e-6)
    rot = HmdMotion(yaw_amp_deg=0.0, pitch_amp_deg=20.0, period_s=8.0).rotation(1.0)
    q = unity_quaternion_to_handumi([rot["x"], rot["y"], rot["z"], rot["w"]])
    yaw, pitch = hmd_yaw_pitch_deg(np.concatenate([[0, 0, 0], q]))
    assert (yaw, pitch) == pytest.approx((0.0, -20.0), abs=1e-6)


def test_straight_down_has_no_yaw():
    assert hmd_yaw_pitch_deg(pose_from_yaw_pitch(0, -90)) is None


def test_wrap_deg():
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-190) == pytest.approx(170)


def test_deadband_is_backlash():
    db = Deadband(1.0)
    db.reset(0.0)
    assert db(0.8) == 0.0
    assert db(2.0) == pytest.approx(1.0)
    assert db(1.5) == pytest.approx(1.0)
    assert db(-1.0) == pytest.approx(0.0)


def test_one_euro_smooths_jitter_and_tracks_steps():
    f = OneEuroFilter(min_cutoff_hz=1.0, beta=0.05, d_cutoff_hz=1.0)
    rng = np.random.default_rng(0)
    out = [f(0.3 * rng.standard_normal(), i / 50) for i in range(200)]
    assert np.std(out[50:]) < 0.1
    for i in range(200, 400):
        last = f(10.0, i / 50)
    assert last == pytest.approx(10.0, abs=0.05)


def test_rate_limiter_respects_velocity_and_acceleration():
    lim = RateLimiter(max_velocity=60.0, max_acceleration=300.0)
    lim.reset(0.0)
    dt, positions, velocities = 0.02, [], []
    for _ in range(200):
        positions.append(lim(30.0, dt))
        velocities.append(lim.velocity)
    v = np.diff([0.0, *positions]) / dt
    assert np.max(np.abs(v)) <= 60.0 + 1e-6
    assert np.max(np.abs(np.diff([0.0, *v]))) / dt <= 300.0 * 1.05 + 1e-6
    assert positions[-1] == pytest.approx(30.0)
    assert max(positions) <= 30.0 + 1e-9  # no overshoot


def test_idle_until_anchored():
    rt = HeadRetargeter(make_config())
    step = rt.step(pose_from_yaw_pitch(10, 0), True, 0.0)
    assert step.state is HeadState.IDLE and step.command_pan_deg is None


def test_anchor_relative_mapping_with_signs_and_clamp():
    rt = HeadRetargeter(make_config())
    # Operator starts looking 50 deg left and 30 deg down: that is the zero.
    assert rt.anchor(pose_from_yaw_pitch(50, -30), (-2.9, 71.8), 0.0)
    step = rt.step(pose_from_yaw_pitch(50, -30), True, 0.02)
    assert (step.command_pan_deg, step.command_tilt_deg) == pytest.approx((-2.9, 71.8))
    step = rt.step(pose_from_yaw_pitch(60, -25), True, 0.04)
    assert step.rel_yaw_deg == pytest.approx(10.0, abs=1e-6)
    assert step.rel_pitch_deg == pytest.approx(5.0, abs=1e-6)
    assert step.command_pan_deg == pytest.approx(7.1, abs=1e-3)
    assert step.command_tilt_deg == pytest.approx(76.8, abs=1e-3)
    step = rt.step(pose_from_yaw_pitch(120, 30), True, 0.06)  # far beyond the window
    assert step.command_pan_deg == pytest.approx(17.1)
    assert step.command_tilt_deg == pytest.approx(86.8)


def test_negative_sign_and_scale():
    cfg = make_config(pan=AxisConfig(home_deg=0.0, range_deg=20.0, sign=-1.0, scale=0.5))
    rt = HeadRetargeter(cfg)
    rt.anchor(pose_from_yaw_pitch(0, 0), (0.0, 71.8), 0.0)
    step = rt.step(pose_from_yaw_pitch(10, 0), True, 0.02)
    assert step.command_pan_deg == pytest.approx(-5.0, abs=1e-4)


def test_disabled_axis_stays_at_anchor():
    rt = HeadRetargeter(make_config().with_axes("pan"))
    rt.anchor(pose_from_yaw_pitch(0, 0), (0.0, 70.0), 0.0)
    step = rt.step(pose_from_yaw_pitch(10, 10), True, 0.02)
    assert step.command_pan_deg == pytest.approx(10.0, abs=1e-4)
    assert step.command_tilt_deg == pytest.approx(70.0)


def test_hmd_loss_holds_last_command_and_resumes_smoothly():
    cfg = make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0)
    rt = HeadRetargeter(cfg)
    rt.anchor(pose_from_yaw_pitch(0, 0), (-2.9, 71.8), 0.0)
    t = 0.0
    for _ in range(100):
        t += 0.02
        step = rt.step(pose_from_yaw_pitch(10, 0), True, t)
    held = step.command_pan_deg
    for _ in range(10):
        t += 0.02
        hold = rt.step(None, False, t)
        assert hold.state is HeadState.HOLD
        assert hold.command_pan_deg == pytest.approx(held)
    t += 0.02
    resumed = rt.step(pose_from_yaw_pitch(-15, 0), True, t)
    assert resumed.state is HeadState.RUNNING
    # Velocity restarts from zero after HOLD: first step moves at most a*dt*dt.
    assert abs(resumed.command_pan_deg - held) <= 300.0 * 0.02 * 0.02 + 1e-9


def test_anchor_clamps_start_into_window():
    rt = HeadRetargeter(make_config())
    rt.anchor(pose_from_yaw_pitch(0, 0), (40.0, 71.8), 0.0)
    step = rt.step(pose_from_yaw_pitch(0, 0), True, 0.02)
    assert step.target_pan_deg == pytest.approx(17.1)
