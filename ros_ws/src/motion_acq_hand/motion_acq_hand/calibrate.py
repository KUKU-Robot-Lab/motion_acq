"""Record the calibration poses for one glove side and write the calibration YAML.

    ros2 run motion_acq_hand calibrate --side right --user op1   # lab glove of that side
    ros2 run motion_acq_hand calibrate --side right --user fake --fake --yes   # fake glove

For each pose in configs/hands/nova2_to_rh56f1.yaml (Korean prompt) the
operator holds still and presses Enter; the tool records --seconds of samples.
A pose that moved, or two poses that came out the same, are asked again
(motion_acq.hand.calibration.run_session) instead of ending the run. Enters
typed while a pose records are dropped, so they cannot record the next pose.
Saves configs/hands/calibration/<user>_<side>.yaml.
--fake switches the fake glove to each pose itself; --yes skips the prompts.
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import argparse
import sys
import termios
import time
from pathlib import Path

from motion_acq_hand.common import (
    REPO_ROOT,
    check_side,
    glove_qos,
    glove_topic,
    load_gloves,
    require_fake_isolation,
)

import rclpy
from rclpy.node import Node
from senseglove_msgs.msg import SenseGloveState
from std_msgs.msg import String

from motion_acq.hand.calibration import CalibrationError, run_session
from motion_acq.hand.nova2 import GloveDataError, angles_from_state, features
from motion_acq.hand.retarget import DEFAULT_RETARGET, load_hand_retarget_config


class Recorder(Node):
    def __init__(self, topic: str, side: str) -> None:
        super().__init__("motion_acq_hand_calibrate")
        self.side = side
        self.samples: list[dict[str, float]] | None = None
        self.last_rx = 0.0
        self.specs = None
        self.create_subscription(SenseGloveState, topic, self._on_glove, glove_qos())
        self.pose_pub = self.create_publisher(String, f"/motion_acq/fake_glove/{side}/pose", 10)

    def _on_glove(self, msg: SenseGloveState) -> None:
        self.last_rx = time.monotonic()
        if self.samples is None:
            return
        try:
            angles = angles_from_state(list(msg.joint_names), list(msg.position), self.side)
        except GloveDataError:
            return
        self.samples.append(features(angles, self.specs))


def spin_for(node: Node, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)


def record_pose(node: Recorder, seconds: float) -> list[dict[str, float]]:
    node.samples = []
    spin_for(node, seconds)
    samples, node.samples = node.samples, None
    if len(samples) < 10:
        raise CalibrationError(f"{seconds} 초에 장갑 샘플 {len(samples)} 개뿐: 장갑 드라이버가 도는지 확인")
    return samples


def ask_enter(text: str) -> None:
    """Wait for a fresh Enter: drop Enters typed earlier (e.g. during a recording)."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    input(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--glove-serial", default=None,
                    help="default: the side's glove in configs/hands/nova2_gloves.yaml (fake: 0)")
    ap.add_argument("--glove-topic", default="")
    ap.add_argument("--retarget-config", type=Path, default=DEFAULT_RETARGET)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--fake", action="store_true", help="drive the fake glove to each pose")
    ap.add_argument("--yes", action="store_true", help="no prompts (fake runs)")
    args = ap.parse_args(argv)
    side = check_side(args.side)
    config = load_hand_retarget_config(args.retarget_config)
    out = args.out or REPO_ROOT / "configs" / "hands" / "calibration" / f"{args.user}_{side}.yaml"
    serial = args.glove_serial or ("0" if args.fake else load_gloves()[side].serial)
    topic = args.glove_topic or glove_topic(serial, side)
    if args.fake:
        require_fake_isolation("calibrate --fake")

    rclpy.init()
    node = Recorder(topic, side)
    node.specs = config.features
    try:
        spin_for(node, 1.0)
        if time.monotonic() - node.last_rx > 0.5:
            raise SystemExit(f"no glove samples on {topic}")

        def record(pose: str) -> list[dict[str, float]]:
            if args.fake:
                node.pose_pub.publish(String(data=pose))
                spin_for(node, 0.5)
            return record_pose(node, args.seconds)

        calibration = run_session(
            side=side, user=args.user, poses=config.poses, feature_poses=config.feature_poses,
            min_span=config.min_span_rad, record=record, say=lambda text: print(text, flush=True),
            ask=(lambda text: None) if args.yes else ask_enter,
        )
        calibration.save(out)
        print(f"보정 저장됨: {out}")
        for name, r in calibration.ranges.items():
            print(f"  {name:17s} open {r.open:+.3f}  closed {r.closed:+.3f} rad")
    except CalibrationError as exc:
        raise SystemExit(f"calibration failed: {exc}") from exc
    finally:
        if args.fake:
            node.pose_pub.publish(String(data="cycle"))
            spin_for(node, 0.2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
