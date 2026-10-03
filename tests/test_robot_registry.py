"""Registry-level embodiment checks (docs/source/development/new_embodiment.md §3).

Currently exercises the ``metal`` embodiment; extend the module-level list as
new simulation-only robots are added.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from motion_acq.robots.registry import (
    CONFIG_DIR,
    available_robot_names,
    load_embodiment,
    robot_config_metadata,
)

EMBODIMENTS = ("openarmv1",)


@pytest.fixture(scope="module", params=EMBODIMENTS)
def runtime(request):
    return load_embodiment(request.param)


def test_registered(runtime) -> None:
    assert runtime.name in available_robot_names()


def test_arm_joint_order_and_indices(runtime) -> None:
    names = list(runtime.robot.joints.actuated_names)
    for side in ("left", "right"):
        arm_names = runtime.arm_joint_names(side)
        indices = runtime.arm_joint_indices(side)
        assert [names[i] for i in indices] == arm_names
        assert all(f"_{side}_" in name for name in arm_names)
    left = set(runtime.arm_joint_indices("left"))
    right = set(runtime.arm_joint_indices("right"))
    assert not left & right
    assert len(left | right) == runtime.robot.joints.num_actuated_joints


def test_home_q_within_limits(runtime) -> None:
    home = runtime.home_q()
    assert home.shape == (runtime.robot.joints.num_actuated_joints,)
    lower = np.asarray(runtime.robot.joints.lower_limits)
    upper = np.asarray(runtime.robot.joints.upper_limits)
    assert np.all(home >= lower - 1e-6)
    assert np.all(home <= upper + 1e-6)


def test_posture_q_defaults_to_home(runtime) -> None:
    """A robot without a declared working posture keeps the legacy behavior."""
    np.testing.assert_allclose(runtime.config.posture_q, runtime.home_q())
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    np.testing.assert_allclose(solver.posture_q, runtime.home_q())


def test_home_fk_is_left_right_symmetric(runtime) -> None:
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    left, right = solver.fk_pose7(runtime.home_q())
    # OpenArm mirrors across the Y=0 world plane: same reach and height.
    np.testing.assert_allclose(left[1], -right[1], atol=1e-3)
    np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=1e-3)


def test_gripper_mapping(runtime) -> None:
    q = runtime.home_q()
    runtime.set_finger_positions(q, {"left": 1.0, "right": 0.0})
    for finger in runtime.finger_joints["left"]:
        assert q[finger.index] == pytest.approx(finger.open_value)
    for finger in runtime.finger_joints["right"]:
        assert q[finger.index] == pytest.approx(finger.closed_value)


def test_urdf_meshes_resolve(runtime) -> None:
    urdf = runtime.load_urdf(load_meshes=True)
    assert len(urdf.scene.geometry) > 0
    for name, geometry in urdf.scene.geometry.items():
        assert geometry.vertices.shape[0] > 0, f"empty mesh for {name}"


def test_collision_model_clean_at_home(runtime) -> None:
    """Structurally-overlapping capsule pairs are auto-ignored at build time."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    distances = np.asarray(
        solver.robot_collision.compute_self_collision_distance(
            runtime.robot, runtime.home_q()
        )
    )
    margin = runtime.config.ik_weights.self_collision_margin
    assert float(distances.min()) >= margin - 1e-6


def test_self_collision_cost_prevents_interpenetration(runtime) -> None:
    """Commanding both TCPs to one point must stop at contact, not merge."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    center = ((left[:3] + right[:3]) / 2).astype(np.float32)
    q = home
    for _ in range(20):
        q = np.asarray(
            solver.ik(
                q,
                left_pose=(center, left[3:7]),
                right_pose=(center, right[3:7]),
            ),
            dtype=np.float32,
        )
    separation = float(
        np.asarray(
            solver.robot_collision.compute_self_collision_distance(runtime.robot, q)
        ).min()
    )
    assert separation > -0.005  # capsules may touch but must not interpenetrate


def test_world_collision_cost_keeps_tcp_above_table(runtime) -> None:
    """Commanding TCPs below z=0 must stop at the tabletop halfspace."""
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    if solver.robot_collision is None:
        pytest.skip("collision not enabled for this embodiment")
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    below_left = (np.array([left[0], left[1], -0.05], dtype=np.float32), left[3:7])
    below_right = (np.array([right[0], right[1], -0.05], dtype=np.float32), right[3:7])
    q = home
    for _ in range(20):
        q = np.asarray(
            solver.ik(q, left_pose=below_left, right_pose=below_right),
            dtype=np.float32,
        )
    l_sol, r_sol = solver.fk_pose7(q)
    assert l_sol[2] > -0.005
    assert r_sol[2] > -0.005


@pytest.mark.parametrize("name", ("openarmv1",))
def test_replay_ik_weights_never_leak_into_the_teleop_profile(name) -> None:
    """Real backends must keep exactly the weights their own YAML declares."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    declared = data["ik_weights"]
    rt = load_embodiment(name)
    assert rt.config.ik_weights.ori_weight == pytest.approx(declared["ori"])
    assert rt.config.ik_weights.rest_weight == pytest.approx(declared["rest"])
    overrides = (data.get("replay") or {}).get("ik_weights") or {}
    assert rt.config.replay_ik_weights == {
        key: float(value) for key, value in overrides.items()
    }


def test_ik_converges_near_home(runtime) -> None:
    solver = runtime.solver_cls(config=runtime.config.ik_weights)
    home = runtime.home_q()
    left, right = solver.fk_pose7(home)
    offset = np.array([0.0, 0.03, -0.03], dtype=np.float32)
    left_target = (left[:3] + offset, left[3:7])
    right_target = (right[:3] + offset, right[3:7])
    q = home
    for _ in range(5):
        q = np.asarray(
            solver.ik(q, left_pose=left_target, right_pose=right_target),
            dtype=np.float32,
        )
    l_sol, r_sol = solver.fk_pose7(q)
    # The rest/posture terms pull OpenArm ~2.7 mm off a 4 cm step target.
    assert float(np.linalg.norm(l_sol[:3] - left_target[0])) < 0.005
    assert float(np.linalg.norm(r_sol[:3] - right_target[0])) < 0.005


def test_robot_config_metadata_identifies_the_profile_it_names() -> None:
    openarm = robot_config_metadata("openarmv1")
    assert openarm["name"] == "openarmv1"
    assert openarm["configuration"]["kind"] == "openarmv1"
    assert len(openarm["sha256"]) == 64
    assert openarm["config_path"].endswith("openarmv1.yaml")
    with pytest.raises(SystemExit, match="Unknown robot"):
        robot_config_metadata("not-a-robot")
