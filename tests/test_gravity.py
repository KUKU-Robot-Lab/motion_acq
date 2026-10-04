"""Gravity feedforward (sim2real model) and the safe start on the fake arm."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from motion_acq.real.openarm.gravity import load_arm_gravity

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets" / "openarm_rh56f1" / "openarm_rh56f1.urdf"
KP = np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0])
HOME = {
    "right": np.array([-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646]),
    "left": np.array([1.2127, -0.2026, -0.6538, 1.7608, -0.3791, -0.5785, -0.6646]),
}
RC_KINEMATICS = Path.home() / "rl_ws/robot_control/src/robot_control/kinematics.py"
HDGP_URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"
S2R_PAYLOAD = {"left": [0.1915800, 0.020847, 0.011958, 0.179547],
               "right": [0.1915800, 0.014897, -0.031431, 0.179734]}  # pd_rh56f1_exec.yaml


def model(side: str):
    return load_arm_gravity(URDF, side, f"{side[0]}_hl_palm_sensor")


def test_pd_only_sag_at_home_matches_the_10_04_right_j7():
    """Without feedforward the wrist sags tau/kp; the real right j7 stopped 0.129 rad short."""
    sag = model("right")(HOME["right"]) / KP
    assert sag[6] == pytest.approx(0.129, abs=0.02)
    assert np.abs(model("right")(np.zeros(7))).max() < 0.1  # hanging at rest: almost nothing


def test_left_is_the_mirror_of_right():
    right, left = model("right")(HOME["right"]), model("left")(HOME["left"])
    assert np.allclose(np.abs(right), np.abs(left), atol=0.05)


@pytest.mark.skipif(not (RC_KINEMATICS.exists() and HDGP_URDF.exists()),
                    reason="robot_control / hdgp not checked out next to this repo")
@pytest.mark.parametrize("side", ["right", "left"])
def test_same_torque_as_the_sim2real_pd_model(side):
    spec = importlib.util.spec_from_file_location("rc_kinematics", RC_KINEMATICS)
    assert spec is not None and spec.loader is not None
    rc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rc
    try:
        spec.loader.exec_module(rc)
    finally:
        sys.modules.pop(spec.name, None)
    prefix = side[0]
    chain = rc.chain_from_urdf(HDGP_URDF.read_text(), [f"{prefix}_aj_{i}" for i in range(1, 8)],
                               f"{prefix}_hl_palm_sensor")
    chain = rc.with_payload(chain, S2R_PAYLOAD[side][0], S2R_PAYLOAD[side][1:])
    ours = model(side)
    rng = np.random.default_rng(1)
    for q in [np.zeros(7), HOME[side], *rng.uniform(-1.5, 1.5, (30, 7))]:
        assert np.allclose(ours.chain.gravity_torque(q), chain.gravity_torque(q), atol=1e-6)


def fake_right(monkeypatch, *, feedforward: bool, home_timeout_s: float = 20.0):
    from motion_acq.real.openarm import driver
    from motion_acq.real.openarm.fake import build_backend
    from motion_acq.robots.registry import load_embodiment, resolve_home_q

    from motion_acq.real.openarm.home_path import HomePath

    real_load = driver.load_home_path

    def every_tenth(path, side):  # same route, 10x fewer samples: a 1.4 s path for the test
        full = real_load(path, side)
        return HomePath(side=side, q=np.vstack([full.q[::10], full.q[-1:]]), dt=full.dt)

    monkeypatch.setattr(driver, "load_home_path", every_tenth)
    monkeypatch.setattr(driver, "MAX_PATH_SPEED_RAD_S", 2.0)  # let the streamer keep up with it
    monkeypatch.delenv("MACQ_FAKE_ROBOT_START", raising=False)  # start at rest
    runtime = load_embodiment("openarm_rh56f1")
    rig = ROOT / "configs" / "stations" / "arm4090.yaml"
    backend = build_backend(runtime=runtime, rig_config=rig, active_sides=("right",))
    env = backend.environment
    changes = {"home_timeout_s": home_timeout_s}
    if not feedforward:
        changes["gravity_urdf"] = ""  # controller without feedforward; the fake arm still sags
    env.settings = dataclasses.replace(env.settings, **changes)
    _, home = resolve_home_q(runtime, rig_config=rig)
    return backend, env, home


def test_fake_arm_reaches_home_with_feedforward(monkeypatch):
    backend, env, home = fake_right(monkeypatch, feedforward=True)
    backend.connect()
    try:
        backend.home(home)
        assert np.abs(env.streamer.feedback()["right"] - HOME["right"]).max() < 0.02
    finally:
        backend.disconnect()


def test_failed_start_returns_to_rest_instead_of_dropping(monkeypatch):
    """10.04: home timed out (j7 sag) and the motors went off at home. Now: back to rest first."""
    backend, env, home = fake_right(monkeypatch, feedforward=False, home_timeout_s=2.0)
    backend.connect()
    try:
        with pytest.raises(TimeoutError, match="joint7"):
            backend.home(home)
        assert np.abs(env.streamer.feedback()["right"]).max() < 0.1  # at rest before the motors go off
    finally:
        backend.disconnect()


def test_unpowered_arm_is_refused():
    """10.04: arm power off -> every joint read exactly 0 -> the path ran into a following error."""
    import functools

    from motion_acq.real.openarm.driver import OpenArmSdkSide
    from motion_acq.real.openarm.fake_sdk import FakeOpenArmSdk

    sdk = FakeOpenArmSdk(start_q_by_port={"can0": [0.0] * 7})
    side = functools.partial(OpenArmSdkSide, sdk=sdk)("can0", enable_fd=True, kp=tuple(KP), kd=(1.0,) * 7,
                                                       gripper_enabled=False)
    with pytest.raises(RuntimeError, match="powered on"):
        side.read_startup_q()


def test_power_off_away_from_rest_holds_and_warns(monkeypatch, caplog):
    backend, env, home = fake_right(monkeypatch, feedforward=True)
    backend.connect()
    backend.home(home)
    monkeypatch.setattr("sys.stdin", None)  # not a terminal: warn, do not wait
    with caplog.at_level("ERROR"):
        backend.disconnect()
    assert "away from rest" in caplog.text and "will drop" in caplog.text


def test_wrist_drift_at_rest_is_aligned_before_the_path(monkeypatch):
    """10.04 third run: j5 6.4, j6 -4.5, j7 4.0 deg after hanging unpowered; align, then the path."""
    backend, env, home = fake_right(monkeypatch, feedforward=True)
    drifted = np.deg2rad([-1.0, 0.6, -2.5, 2.0, 6.4, -4.5, 4.0]).tolist()
    backend.fake_sdk.start_q_by_port[env.settings.right_port] = drifted
    backend.connect()
    try:
        backend.home(home)
        assert np.abs(env.streamer.feedback()["right"] - HOME["right"]).max() < 0.02
    finally:
        backend.disconnect()
