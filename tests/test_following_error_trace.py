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
        saved = sorted(tmp_path.glob("following_error_right_*.npz"))
        assert len(saved) == 1 and str(saved[0]) in message
        data = np.load(saved[0])
        assert len(data["t"]) >= 20
        gap = np.abs(data["right_commanded"][-1, 6] - data["right_measured"][-1, 6])
        assert gap == pytest.approx(0.35, abs=0.05)
    finally:
        monkeypatch.setattr("sys.stdin", None)  # power-off prompt: warn, do not wait
        try:
            backend.disconnect()
        except RuntimeError:
            pass
