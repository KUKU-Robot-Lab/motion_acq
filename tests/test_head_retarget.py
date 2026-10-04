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


def test_near_vertical_has_no_yaw_but_keeps_pitch():
    yaw, pitch = hmd_yaw_pitch_deg(pose_from_yaw_pitch(0, -90))
    assert yaw is None and pitch == pytest.approx(-90.0)
    yaw, pitch = hmd_yaw_pitch_deg(pose_from_yaw_pitch(30, 80))
    assert yaw is None and pitch == pytest.approx(80.0)
    assert hmd_yaw_pitch_deg(pose_from_yaw_pitch(30, 70))[0] == pytest.approx(30.0)


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
    t = 0.04
    for k in range(1, 61):  # turn on gradually (1 deg/frame) far beyond the window
        t += 0.02
        step = rt.step(pose_from_yaw_pitch(60 + k, -25 + k), True, t)
    assert not step.reanchored
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


def _walk(rt, start_t, poses, dt=0.02):
    t, steps = start_t, []
    for pose in poses:
        t += dt
        steps.append(rt.step(pose, True, t))
    return t, steps


def test_yaw_unwraps_across_180_without_flipping():
    rt = HeadRetargeter(make_config())
    rt.anchor(pose_from_yaw_pitch(170, 0), (-2.9, 71.8), 0.0)
    # Turn left through +180 (raw yaw wraps to -180..-170): relative +20.
    path = [pose_from_yaw_pitch(170 + k, 0) for k in range(1, 21)]
    _, steps = _walk(rt, 0.0, path)
    assert steps[-1].rel_yaw_deg == pytest.approx(20.0, abs=1e-6)
    assert all(b.rel_yaw_deg > a.rel_yaw_deg for a, b in zip(steps, steps[1:], strict=False))


def test_jitter_opposite_the_anchor_stays_on_one_side():
    rt = HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0))
    rt.anchor(pose_from_yaw_pitch(0, 0), (-2.9, 71.8), 0.0)
    walk = [pose_from_yaw_pitch(k, 0) for k in range(1, 180)]  # turn left to 179
    t, _ = _walk(rt, 0.0, walk)
    jitter = [pose_from_yaw_pitch(179 if i % 2 else -179, 0) for i in range(100)]
    _, steps = _walk(rt, t, jitter)
    targets = {round(s.target_pan_deg, 3) for s in steps}
    assert targets == {17.1}  # pinned at the left end, never the right end


def test_yaw_is_held_while_looking_near_vertically():
    rt = HeadRetargeter(make_config())
    rt.anchor(pose_from_yaw_pitch(0, 60), (-2.9, 71.8), 0.0)
    path = [pose_from_yaw_pitch(0, 60 + k) for k in range(1, 26)]  # up to 85 deg
    t, steps = _walk(rt, 0.0, path)
    noisy = [pose_from_yaw_pitch(25 * (-1) ** i, 85) for i in range(20)]
    _, steps = _walk(rt, t, noisy)
    assert {round(s.rel_yaw_deg, 6) for s in steps} == {0.0}
    assert steps[-1].rel_pitch_deg == pytest.approx(25.0)


def test_anchor_refused_while_looking_vertically():
    rt = HeadRetargeter(make_config())
    assert not rt.anchor(pose_from_yaw_pitch(0, 88), (-2.9, 71.8), 0.0)
    assert rt.state is HeadState.IDLE


def test_long_hold_reanchors_without_moving_the_head():
    rt = HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0))
    rt.anchor(pose_from_yaw_pitch(0, 0), (-2.9, 71.8), 0.0)
    t, steps = _walk(rt, 0.0, [pose_from_yaw_pitch(10, 0)] * 100)
    held = steps[-1].command_pan_deg
    for _ in range(50):  # 1 s lost: the operator turned, or the app restarted
        t += 0.02
        rt.step(None, False, t)
    step = rt.step(pose_from_yaw_pitch(-20, 0), True, t + 0.02)
    assert step.reanchored and step.command_pan_deg == pytest.approx(held)
    t, steps = _walk(rt, t + 0.02, [pose_from_yaw_pitch(-15, 0)] * 100)
    assert steps[-1].command_pan_deg == pytest.approx(held + 5.0, abs=0.05)


def test_recenter_jump_reanchors_instead_of_snapping():
    rt = HeadRetargeter(make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0))
    rt.anchor(pose_from_yaw_pitch(0, 0), (-2.9, 71.8), 0.0)
    t, steps = _walk(rt, 0.0, [pose_from_yaw_pitch(5, 0)] * 100)
    before = steps[-1].command_pan_deg
    step = rt.step(pose_from_yaw_pitch(95, 0), True, t + 0.02)  # Quest recenter
    assert step.reanchored and step.command_pan_deg == pytest.approx(before)


def test_stalled_loop_does_not_turn_into_a_big_step():
    cfg = make_config(max_velocity_deg_s=60.0, max_acceleration_deg_s2=300.0)
    rt = HeadRetargeter(cfg)
    rt.anchor(pose_from_yaw_pitch(0, 0), (-2.9, 71.8), 0.0)
    t, steps = _walk(rt, 0.0, [pose_from_yaw_pitch(k, 0) for k in range(1, 16)])
    before = steps[-1].command_pan_deg
    step = rt.step(pose_from_yaw_pitch(15, 0), True, t + 2.0)  # 2 s stall
    assert abs(step.command_pan_deg - before) <= 60.0 * cfg.max_step_dt_s + 1e-6


def test_rate_limiter_target_reversal_keeps_limits():
    lim = RateLimiter(max_velocity=60.0, max_acceleration=300.0)
    lim.reset(0.0)
    dt, positions = 0.02, []
    for i in range(300):
        positions.append(lim(20.0 if i < 30 else -20.0, dt))
    v = np.diff([0.0, *positions]) / dt
    a = np.diff([0.0, *v]) / dt
    assert np.max(np.abs(v)) <= 60.0 + 1e-6
    # Every step keeps the limit except the single arrival step, where the snap
    # onto the target leaves a quantisation residue (measured 1.11x).
    arrivals = [int(np.argmax(np.isclose(positions, x))) for x in (20.0, -20.0)]
    others = np.delete(np.abs(a), [i + k for i in arrivals for k in (0, 1)])
    assert np.max(others) <= 300.0 + 1e-6
    assert np.max(np.abs(a)) <= 300.0 * 1.15
    assert min(positions) >= -20.0 - 1e-9 and positions[-1] == pytest.approx(-20.0)



def test_asymmetric_window_follows_the_sign():
    """arm4090 tilt: sign -1 (encoder + looks down), 30 deg up, 15 deg down."""
    from motion_acq.head.retarget import AxisConfig

    tilt = AxisConfig(home_deg=71.8, range_deg=15.0, sign=-1.0, range_pos_deg=30.0)
    assert (tilt.lower_deg, tilt.upper_deg) == pytest.approx((41.8, 86.8))
    pan = AxisConfig(home_deg=0.0, range_deg=20.0, sign=1.0, range_pos_deg=25.0, range_neg_deg=10.0)
    assert (pan.lower_deg, pan.upper_deg) == pytest.approx((-10.0, 25.0))
    with pytest.raises(ValueError):
        AxisConfig(home_deg=0.0, range_deg=20.0, range_neg_deg=0.0)
