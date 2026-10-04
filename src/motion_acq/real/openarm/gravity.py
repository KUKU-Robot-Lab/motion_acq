"""Gravity torque feedforward for the OpenArm MIT commands, as sim2real does it.

Port of robot_control/src/robot_control/kinematics.py (chain_from_urdf, the
fixed-link lumping and Chain.gravity_torque), the model sim2real's pd_arm uses
for RH56F1 arms (pd_rh56f1_exec.yaml gravity: model_tau_ff, tip
<r|l>_hl_palm_sensor, scale 1, cap 20 N m).

Without it the arm is held by PD alone (kp 10 on the wrist joints) and sags
under the RH56F1: on 10.04 the right j7 stopped 0.129 rad short of home and
the home step timed out (sim2real saw j7 0.21 rad on 09.23 before it added
this feedforward).

sim2real loads the hdgp URDF, whose hand joints move, and adds the hand
beyond them as a payload computed at the open hand. motion_acq's
openarm_rh56f1 URDF has the hand fixed at that same open pose, so the plain
fixed-link lumping already carries the whole hand; tests/test_gravity.py
checks that both give the same torque.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.80665])  # along -z of the URDF root


class GravityModelError(ValueError):
    pass


@dataclass(frozen=True)
class Revolute:
    name: str
    axis: np.ndarray
    origin: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class Link:
    name: str
    mass: float
    com: np.ndarray  # in the link's own frame


class Chain:
    """Serial chain of revolute joints with each driven link's lumped mass."""

    def __init__(self, joints: Sequence[Revolute], links: Sequence[Link]) -> None:
        if len(joints) != len(links):
            raise GravityModelError(f"{len(joints)} joints but {len(links)} links")
        self.joints = tuple(joints)
        self.links = tuple(links)

    def frames(self, q: Sequence[float]) -> list[np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joints),):
            raise GravityModelError(f"chain has {len(self.joints)} joints, got {q.size} values")
        frames, current = [], np.eye(4)
        for joint, angle in zip(self.joints, q, strict=True):
            step = np.eye(4)
            step[:3, :3] = joint.rotation @ _rotation(joint.axis, float(angle))
            step[:3, 3] = joint.origin
            current = current @ step
            frames.append(current)
        return frames

    def gravity_torque(self, q: Sequence[float]) -> np.ndarray:
        """Joint torques that hold the chain against gravity (masses and centres only)."""
        frames = self.frames(q)
        torque = np.zeros(len(self.joints))
        centres = [frame[:3, :3] @ link.com + frame[:3, 3] for frame, link in zip(frames, self.links, strict=True)]
        for index, (joint, frame) in enumerate(zip(self.joints, frames, strict=True)):
            axis = frame[:3, :3] @ joint.axis
            for link, centre in zip(self.links[index:], centres[index:], strict=True):
                lever = np.cross(axis, centre - frame[:3, 3])
                torque[index] -= link.mass * float(GRAVITY @ lever)
        return torque


def chain_from_urdf(urdf: str, joint_names: Iterable[str], tip_link: str) -> Chain:
    """Chain for joint_names ending at tip_link; fixed joints fold into neighbours."""
    root = ElementTree.fromstring(urdf)
    joints = {e.get("name"): e for e in root.findall("joint")}  # direct children only
    links = {e.get("name"): e for e in root.findall("link")}
    wanted = list(joint_names)
    missing = [name for name in wanted if name not in joints]
    if missing:
        raise GravityModelError(f"URDF has no joint named {missing[0]!r}")
    if tip_link not in links:
        raise GravityModelError(f"URDF has no link named {tip_link!r}")
    parent_of = {}
    for element in joints.values():
        child = element.find("child")
        if child is not None:
            parent_of[child.get("link")] = element
    path, link = [], tip_link
    while link in parent_of:
        element = parent_of[link]
        path.append(element)
        link = element.find("parent").get("link")
    path.reverse()
    on_path = [e.get("name") for e in path]
    unreachable = [name for name in wanted if name not in on_path]
    if unreachable:
        raise GravityModelError(f"joint {unreachable[0]!r} is not between the URDF root and {tip_link!r}")
    fixed_children: dict[str, list[ElementTree.Element]] = {}
    for element in joints.values():
        if element.get("type") == "fixed" and element.find("parent") is not None:
            fixed_children.setdefault(element.find("parent").get("link"), []).append(element)
    chain_joints, chain_links, pending = [], [], np.eye(4)
    for element in path:
        origin, rotation = _origin(element)
        if element.get("name") not in wanted:
            pending = pending @ _homogeneous(rotation, origin)
            continue
        combined = pending @ _homogeneous(rotation, origin)
        child = element.find("child").get("link")
        chain_joints.append(Revolute(element.get("name"), _axis(element), combined[:3, 3], combined[:3, :3]))
        chain_links.append(_lumped(child, links, fixed_children))
        pending = np.eye(4)
    return Chain(chain_joints, chain_links)


def _lumped(name, links, fixed_children, transform=None) -> Link:
    transform = np.eye(4) if transform is None else transform
    element = links.get(name)
    own = _link(element) if element is not None else Link(name, 0.0, np.zeros(3))
    mass = own.mass
    moment = own.mass * (transform[:3, :3] @ own.com + transform[:3, 3])
    for joint in fixed_children.get(name, ()):
        origin, rotation = _origin(joint)
        child = _lumped(joint.find("child").get("link"), links, fixed_children,
                        transform @ _homogeneous(rotation, origin))
        mass += child.mass
        moment += child.mass * child.com
    return Link(name, mass, moment / mass if mass > 0.0 else np.zeros(3))


def _link(element) -> Link:
    inertial = element.find("inertial")
    if inertial is None:
        return Link(element.get("name"), 0.0, np.zeros(3))
    mass = inertial.find("mass")
    origin, _ = _origin(inertial)
    return Link(element.get("name"), 0.0 if mass is None else float(mass.get("value", 0.0)), origin)


def _origin(element) -> tuple[np.ndarray, np.ndarray]:
    origin = element.find("origin")
    if origin is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
    return xyz, _rpy(*rpy)


def _axis(element) -> np.ndarray:
    axis = element.find("axis")
    values = np.array([1.0, 0.0, 0.0]) if axis is None else np.array(
        [float(v) for v in axis.get("xyz", "1 0 0").split()])
    norm = np.linalg.norm(values)
    if norm == 0.0:
        raise GravityModelError(f"joint {element.get('name')!r} has a zero axis")
    return values / norm


def _homogeneous(rotation, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sin * cross + (1.0 - cos) * (cross @ cross)


@dataclass(frozen=True)
class ArmGravity:
    """tau_ff(q) for one OpenArm side, clipped to +-cap_nm per joint."""

    chain: Chain
    scale: np.ndarray
    cap_nm: float

    def __call__(self, q: Sequence[float]) -> np.ndarray:
        tau = self.chain.gravity_torque(q) * self.scale
        return np.clip(tau, -self.cap_nm, self.cap_nm)


def load_arm_gravity(urdf_path: Path, side: str, tip_link: str, *,
                     scale: Sequence[float] = (1.0,) * 7, cap_nm: float = 20.0) -> ArmGravity:
    joints = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    chain = chain_from_urdf(Path(urdf_path).read_text(), joints, tip_link)
    scale_arr = np.asarray(scale, dtype=float)
    if scale_arr.shape != (7,) or not cap_nm > 0:
        raise GravityModelError(f"gravity scale {scale_arr.shape} / cap {cap_nm} invalid")
    return ArmGravity(chain=chain, scale=scale_arr, cap_nm=float(cap_nm))
