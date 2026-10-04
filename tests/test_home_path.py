"""Stored rest <-> home paths (sim2real): loading, start classification, fake playback."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from motion_acq.real.openarm.home_path import (
    HomePathError,
    classify_start,
    load_home_path,
)
from motion_acq.robots.registry import load_embodiment, resolve_home_q

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "configs" / "paths"
S2R_PATHS = Path.home() / "rl_ws/sim2real/deploy/policy_control/paths"


def home_of(side: str) -> np.ndarray:
    runtime = load_embodiment("openarm_rh56f1")
    _, home = resolve_home_q(runtime, rig_config=ROOT / "configs" / "stations" / "arm4090.yaml")
    names = list(runtime.joint_names)
    return np.array([home[names.index(f"openarm_{side}_joint{i}")] for i in range(1, 8)])


@pytest.mark.parametrize("side", ["right", "left"])
def test_paths_go_from_rest_to_the_station_home(side):
    path = load_home_path(PATHS / f"home_rh56f1_{side}.npz", side)
    assert np.allclose(path.rest, 0.0)
    assert np.allclose(path.home, home_of(side), atol=1e-3)
    assert 10.0 < path.duration_s < 20.0
    assert np.abs(np.diff(path.q, axis=0)).max() / path.dt <= 0.31


def test_start_is_rest_or_home_and_nothing_else():
    path = load_home_path(PATHS / "home_rh56f1_right.npz", "right")
    home = home_of("right")
    assert classify_start(np.full(7, 0.03), path, home, 0.10) == "rest"
    drifted = np.deg2rad([-1.0, 0.6, -2.5, 2.0, 6.4, -4.5, 4.0])  # 10.04 right arm, power off at rest
    assert classify_start(drifted, path, home, 0.10) == "near_rest"
    assert classify_start(home + 0.05, path, home, 0.10) == "home"
    with pytest.raises(HomePathError, match="neither"):
        classify_start(home * 0.5, path, home, 0.10)
    with pytest.raises(HomePathError, match="replan"):
        classify_start(np.zeros(7), path, home + 0.2, 0.10)


def test_wrong_side_file_is_refused():
    with pytest.raises(HomePathError, match="expected"):
        load_home_path(PATHS / "home_rh56f1_right.npz", "left")


@pytest.mark.skipif(not S2R_PATHS.exists(), reason="sim2real not checked out next to this repo")
@pytest.mark.parametrize("side", ["right", "left"])
def test_paths_match_sim2real(side):
    ours = (PATHS / f"home_rh56f1_{side}.npz").read_bytes()
    theirs = (S2R_PATHS / f"home_rh56f1_{side}.npz").read_bytes()
    assert hashlib.sha256(ours).hexdigest() == hashlib.sha256(theirs).hexdigest(), \
        "sim2real replanned the path: copy it to configs/paths and update the README"


def test_fake_right_arm_goes_rest_home_rest(monkeypatch):
    """Production driver on the fake SDK: path forward at start, backwards at the end."""
    from motion_acq.real.openarm import driver
    from motion_acq.real.openarm.fake import build_backend

    monkeypatch.setattr(driver.time, "sleep", lambda s: None)  # play the path instantly
    monkeypatch.delenv("MACQ_FAKE_ROBOT_START", raising=False)  # fake motors start at rest
    runtime = load_embodiment("openarm_rh56f1")
    rig = ROOT / "configs" / "stations" / "arm4090.yaml"
    backend = build_backend(runtime=runtime, rig_config=rig, active_sides=("right",))
    _, home = resolve_home_q(runtime, rig_config=rig)
    env = backend.environment
    env.settings = type(env.settings)(**{**env.settings.__dict__, "home_timeout_s": 20.0})
    seen = []
    original = env._play_paths

    def spy(paths, dt, label):
        seen.append((label, {s: (np.round(p[0], 3).tolist(), np.round(p[-1], 3).tolist()) for s, p in paths.items()}))
        return original(paths, dt, label)

    monkeypatch.setattr(env, "_play_paths", spy)
    backend.connect()
    try:
        backend.home(home)
        assert seen[0][0] == "rest -> home"
        assert np.allclose(env.streamer.feedback()["right"], home_of("right"), atol=0.1)
        backend.move_home(home)
        backend.rest(home)
        assert seen[-1][0] == "home -> rest"
        assert np.abs(env.streamer.feedback()["right"]).max() < 0.1
    finally:
        backend.disconnect()
    assert os.environ.get("MACQ_FAKE_ROBOT_START") is None
