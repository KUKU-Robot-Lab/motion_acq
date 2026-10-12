"""A following error says what was commanded and leaves the last seconds on disk.

10.04 21:24:17 the right arm stopped 4 s after Space on "joint7 following error
0.351 rad exceeds 0.350 rad" and nothing else: no commanded or measured angle,
no record of what the arm was asked to do.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from test_gravity import fake_right

from motion_acq.real.openarm import trace
from motion_acq.real.openarm.trace import CommandTrace


def test_trace_keeps_only_the_last_ticks(tmp_path):
    ring = CommandTrace(ticks=3)
    for i in range(5):
        ring.add(float(i), {"right": np.full(7, i)}, {"right": np.full(7, -i)}, {"right": np.full(7, 2 * i)})
    path = ring.dump(tmp_path / "t.npz")
    data = np.load(path)
    assert data["t"].tolist() == [2.0, 3.0, 4.0]
    assert data["right_commanded"][:, 0].tolist() == [2, 3, 4]
    assert data["right_measured"][:, 0].tolist() == [-2, -3, -4]
    assert data["right_target"][:, 0].tolist() == [4, 6, 8]


def test_frozen_reading_is_timed_per_joint():
    ring = CommandTrace(ticks=50)
    for i in range(30):
        moving = np.full(7, 0.01 * i, dtype=np.float32)
        moving[6] = 0.5 if i >= 10 else 0.01 * i  # joint7 stops answering at tick 10
        ring.add(i * 0.01, {"left": moving}, {"left": moving}, {"left": moving})
    assert ring.frozen_s("left", 6) == pytest.approx(0.19)
    assert ring.frozen_s("left", 0) == 0.0
    assert ring.frozen_s("right", 6) == 0.0


def test_a_failing_motor_report_does_not_hide_the_fault():
    from motion_acq.real.openarm.driver import motor_state

    class Broken:
        def motor_state(self, joint):
            raise RuntimeError("no reply")

    assert motor_state(object(), 6) is None  # sides without a report
    assert motor_state(Broken(), 6).startswith("unreadable (RuntimeError")


def test_following_error_names_the_angles_and_saves_the_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path)
    backend, env, home = fake_right(monkeypatch, feedforward=True)
    backend.connect()
    try:
        backend.home(home)
        port = env.settings.right_port
        backend.fake_sdk.arms[port].speed = 0.0  # the arm stops moving (stuck joint, collision)
        target = env.streamer.feedback()["right"].copy()
        target[6] += 0.8
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and env.streamer.error is None:
            env.streamer.set_targets({"right": target})
            time.sleep(0.02)
        error = env.streamer.error
        assert isinstance(error, RuntimeError)
        message = str(error)
        assert "joint7" in message and "commanded" in message and "measured" in message
        assert "has not changed for" in message  # 10.11: a stuck joint says so
        assert "Motor: MOS 30 C, rotor 30 C" in message
        saved = sorted(tmp_path.glob("following_error_right_*.npz"))
        assert len(saved) == 1 and str(saved[0]) in message
        data = np.load(saved[0])
        assert len(data["t"]) >= 20
        gap = np.abs(data["right_commanded"][-1, 6] - data["right_measured"][-1, 6])
        assert gap == pytest.approx(0.35, abs=0.05)
        # 10.12 user: stop pushing; the motors now hold the measured pose, not the blocked target
        motors = backend.fake_sdk.arms[port]._arm.get_motors()
        assert max(abs(m.target - m.position) for m in motors) < 1e-4
    finally:
        monkeypatch.setattr("sys.stdin", None)  # power-off prompt: warn, do not wait
        try:
            backend.disconnect()
        except RuntimeError:
            pass


def test_a_blocked_arm_is_released_then_goes_home_on_exit(monkeypatch, tmp_path, caplog):
    """10.12 user: blocked by the table the arm stopped and kept pressing; now it lets go and goes home."""
    import logging

    from motion_acq.teleop.common import return_home_on_exit

    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path)
    backend, env, home = fake_right(monkeypatch, feedforward=True)
    backend.connect()
    try:
        backend.home(home)
        arm = backend.fake_sdk.arms[env.settings.right_port]
        speed, arm.speed = arm.speed, 0.0  # the table
        target = env.streamer.feedback()["right"].copy()
        target[0] -= 0.8
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and env.streamer.error is None:
            env.streamer.set_targets({"right": target})
            time.sleep(0.02)
        assert env.streamer.error is not None
        arm.speed = speed  # the operator's hand is off the table
        with caplog.at_level("INFO"):
            return_home_on_exit(backend, home, logging.getLogger("test"))
        assert "restarting and returning home" in caplog.text and "Arms at home." in caplog.text
        assert env.streamer.error is None
        motors = arm._arm.get_motors()
        assert max(abs(m.position) for m in motors) < 0.1  # home, then the stored path to rest
    finally:
        monkeypatch.setattr("sys.stdin", None)
        try:
            backend.disconnect()
        except RuntimeError:
            pass
