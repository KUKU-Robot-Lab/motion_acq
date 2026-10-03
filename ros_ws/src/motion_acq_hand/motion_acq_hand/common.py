"""Shared bits for the hand nodes: motion_acq import path, topics, QoS."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_motion_acq_on_path() -> Path:
    """Make ``motion_acq`` importable on the system ROS interpreter.

    Order: $MOTION_ACQ_SRC, then the source tree this file lives in
    (colcon --symlink-install keeps the link to ros_ws/src/...).
    """
    candidates = []
    if os.environ.get("MOTION_ACQ_SRC"):
        candidates.append(Path(os.environ["MOTION_ACQ_SRC"]))
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "motion_acq" / "__init__.py").exists():
            candidates.append(parent / "src")
            break
    for path in candidates:
        if (path / "motion_acq" / "__init__.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path.parent
    raise RuntimeError(
        "motion_acq source not found: set MOTION_ACQ_SRC=<repo>/src or build with --symlink-install"
    )


REPO_ROOT = ensure_motion_acq_on_path()

SIDE_TAG = {"right": "rh", "left": "lh"}


def glove_topic(serial: str, side: str) -> str:
    """senseglove_ros namespace: /senseglove/glove<serial>/<rh|lh>/senseglove_states."""
    return f"/senseglove/glove{serial}/{SIDE_TAG[side]}/senseglove_states"


def hand_ns(side: str) -> str:
    return f"/hand_{side}"


def glove_qos():
    """senseglove_state_broadcaster publishes best_effort; match it."""
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)


def check_side(side: str) -> str:
    if side not in SIDE_TAG:
        raise SystemExit(f"side must be right or left, not {side!r}")
    return side


FAKE_DOMAIN_DEFAULT = "177"


def require_fake_isolation(what: str) -> None:
    """Fake nodes publish the real driver topic names (/hand_<side>/angle_set).

    They may only run localhost-only on the dedicated fake domain, so they can
    never reach a real RH56F1 driver or a real glove stack on any domain.
    """
    want = os.environ.get("MACQ_FAKE_DOMAIN", FAKE_DOMAIN_DEFAULT)
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    localhost = os.environ.get("ROS_LOCALHOST_ONLY", "")
    if domain != want or localhost != "1":
        raise SystemExit(
            f"{what} refuses to start: needs ROS_DOMAIN_ID={want} and ROS_LOCALHOST_ONLY=1 "
            f"(have ROS_DOMAIN_ID={domain or 'unset'}, ROS_LOCALHOST_ONLY={localhost or 'unset'})"
        )


def spin_node(factory) -> None:
    """Create, spin and tear down a node; Ctrl+C/SIGINT (also a repeated one) is a clean stop."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()
