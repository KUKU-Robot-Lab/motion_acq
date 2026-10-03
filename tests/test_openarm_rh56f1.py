"""arm4090 embodiment: OpenArm + RH56F1 built from the hdgp asset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motion_acq.config import station_rig_config
from motion_acq.real.openarm.driver import load_openarm_settings
from motion_acq.real.registry import REAL_ROBOT_NAMES
from motion_acq.robots.registry import load_embodiment, resolve_home_q

ROOT = Path(__file__).resolve().parents[1]
HDGP_URDF = Path.home() / "rl_ws/hdgp/assets/robot/openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.urdf"


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment("openarm_rh56f1")


def test_registered_with_real_backend(runtime):
    assert "openarm_rh56f1" in REAL_ROBOT_NAMES and "openarmv1" in REAL_ROBOT_NAMES
    assert runtime.config.real_options["backend"] == "openarm_can"
    assert runtime.config.real_options["gripper"]["enabled"] is False


def test_arm_joints_only_and_named_for_the_driver(runtime):
    assert len(runtime.joint_names) == 14
    for side in ("left", "right"):
        names = runtime.arm_joint_names(side)
        assert names == [f"openarm_{side}_joint{i}" for i in range(1, 8)]


def test_home_is_sim2real_home_and_posture(runtime):
    home = runtime.home_q()
    names = list(runtime.joint_names)
    right = [home[names.index(f"openarm_right_joint{i}")] for i in range(1, 8)]
    assert right == pytest.approx([-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646], abs=1e-6)
    assert np.allclose(runtime.config.posture_q, home)
    lower = np.asarray(runtime.robot.joints.lower_limits)
    upper = np.asarray(runtime.robot.joints.upper_limits)
    assert np.all((home >= lower) & (home <= upper))


def test_tcp_is_the_palm_sensor_at_home(runtime):
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    left, right = solver.fk_pose7(runtime.home_q())
    # sim2real: right palm_sensor at (0.08, -0.30, 0.45) in this home pose.
    assert right[:3] == pytest.approx([0.08, -0.30, 0.45], abs=2e-3)
    # The RH56F1 asset is ~3 mm asymmetric left/right (thumb module), not mirrored exactly.
    assert left[:3] == pytest.approx([0.08, 0.30, 0.45], abs=5e-3)


def test_ik_converges_near_home(runtime):
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    offset = np.array([0.03, 0.0, -0.02], dtype=np.float32)
    targets = ((left[:3] + offset, left[3:7]), (right[:3] + offset, right[3:7]))
    q = home
    for _ in range(5):
        q = np.asarray(solver.ik(q, left_pose=targets[0], right_pose=targets[1]), dtype=np.float32)
    l_sol, r_sol = solver.fk_pose7(q)
    assert float(np.linalg.norm(l_sol[:3] - targets[0][0])) < 5e-3
    assert float(np.linalg.norm(r_sol[:3] - targets[1][0])) < 5e-3


def test_station_arm4090_selects_it():
    rig = station_rig_config("arm4090")
    runtime = load_embodiment("openarm_rh56f1")
    name, _ = resolve_home_q(runtime, rig_config=rig)
    assert name == "rh56f1_aglt"
    settings = load_openarm_settings(rig, runtime.config.real_options, None, robot_name="openarm_rh56f1")
    assert (settings.left_port, settings.right_port, settings.gripper_enabled) == ("can1", "can0", False)


@pytest.mark.skipif(not HDGP_URDF.exists(), reason="hdgp asset not present")
def test_committed_urdf_matches_a_fresh_build():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_openarm_rh56f1_urdf.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
