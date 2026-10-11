"""One operator command ([홈] / [따라가기] / [주먹] / auto) to one hand node, delivered for sure.

10.11: `ros2 topic pub --once` published as soon as any one subscriber matched and then quit. With the
rosbag recorder also on /motion_acq/hand_<side>/command, the recorder could match first, so the message
went to the bag only, and the left [홈] was dropped twice while the hand kept following the glove. This
tool waits until the hand node itself subscribes, then resends the command until the node's status shows
it (a repeat is harmless: the node's command() does nothing again for a command it already has).

    ros2 run motion_acq_hand hand_command --side left home
"""

# ruff: noqa: I001  -- motion_acq_hand.common must be imported first (it puts motion_acq on sys.path)
from __future__ import annotations

import argparse
import json
import sys
import time

from motion_acq_hand.common import check_side

KINDS = ("home", "follow", "fist", "auto")
RESEND_S = 0.25


def node_name(side: str) -> str:
    return f"motion_acq_hand_{side}"


def acknowledged(status: str, kind: str) -> bool:
    """The hand node's status JSON shows this command (auto: no manual command)."""
    try:
        manual = json.loads(status).get("manual")
    except (ValueError, AttributeError):
        return False
    return manual is None if kind == "auto" else manual == kind


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True)
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="give up after this many seconds (a hand node just launched needs a few)")
    args, _ = ap.parse_known_args(argv)
    side = check_side(args.side)

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node(f"motion_acq_hand_command_{side}")
    topic = f"/motion_acq/hand_{side}/command"
    pub = node.create_publisher(String, topic, 10)
    statuses: list[str] = []

    def on_status(msg: String) -> None:
        statuses.append(msg.data)

    node.create_subscription(String, f"/motion_acq/hand_{side}/status", on_status, 10)
    deadline = time.monotonic() + args.timeout
    sent = 0
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            listening = any(i.node_name == node_name(side) for i in node.get_subscriptions_info_by_topic(topic))
            if not listening:
                continue
            if sent and statuses and acknowledged(statuses[-1], args.kind):
                print(f"{side}: {args.kind} acknowledged by {node_name(side)} (sent {sent}x)", flush=True)
                return 0
            pub.publish(String(data=args.kind))
            sent += 1
            statuses.clear()  # only a status after this send counts
            end = time.monotonic() + RESEND_S
            while time.monotonic() < end:  # let the node take it and publish a status or two
                rclpy.spin_once(node, timeout_sec=0.05)
        why = (f"{node_name(side)} not subscribed to {topic} (is the hand node running?)" if not sent
               else f"sent {sent}x, status never showed manual={args.kind}")
        print(f"{side}: {args.kind} NOT delivered: {why}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
