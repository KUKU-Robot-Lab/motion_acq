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

from motion_acq.cpu import keep_off_rt  # noqa: E402
from motion_acq.hand.nova2 import SIDE_TAG, GloveDataError, check_serial, glove_topic, load_gloves  # noqa: E402,F401


def declare_glove_topic(node, side: str) -> str:
    """glove_topic param, else the topic of the glove_serial param.

    The serial is read raw: -p glove_serial:=00782 arrives as 782.0, which
    must fail here rather than subscribe to a topic nobody publishes.
    """
    from rcl_interfaces.msg import ParameterDescriptor

    topic = str(declare(node, "glove_topic", ""))
    if topic:
        return topic
    serial = node.declare_parameter("glove_serial", "0", ParameterDescriptor(dynamic_typing=True)).value
    try:
        return glove_topic(serial, side)
    except GloveDataError as exc:
        raise SystemExit(str(exc)) from exc


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


# Fixed on purpose (no override): fakes publish the real driver topic names.
FAKE_DOMAIN = "177"


def fake_isolated() -> bool:
    return os.environ.get("ROS_DOMAIN_ID") == FAKE_DOMAIN and os.environ.get("ROS_LOCALHOST_ONLY") == "1"


def require_fake_isolation(what: str) -> None:
    """Fake nodes may only run localhost-only on the dedicated fake domain.

    They publish /hand_<side>/angle_set, angle_actual and the glove topic, so
    on any shared domain they could reach a real RH56F1 driver or hand node.
    """
    if not fake_isolated():
        raise SystemExit(
            f"{what} refuses to start: needs ROS_DOMAIN_ID={FAKE_DOMAIN} and ROS_LOCALHOST_ONLY=1 "
            f"(have ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID') or 'unset'}, "
            f"ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY') or 'unset'})"
        )


def declare(node, name: str, default):
    """Declare a parameter that accepts any type (1234 vs "1234", 1 vs 1.0) and cast it.

    Plain declare_parameter fixes the type from the default, so
    -p glove_serial:=1234 or -p dropout_s:=1 would fail at start on Humble.
    """
    from rcl_interfaces.msg import ParameterDescriptor

    value = node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return type(default)(value)


def spin_node(factory) -> None:
    """Create, spin and tear down a node; Ctrl+C/SIGINT (also a repeated one) is a clean stop."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    print(keep_off_rt(), flush=True)  # off the RH56F1 EtherCAT cores (sim2real cpu plan)
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
