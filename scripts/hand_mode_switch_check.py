#!/usr/bin/env python3
"""RH56F1 runtime finger mode switch, one finger against an object (real hand, run with approval).

10.06 position + force loop (motion_acq.hand.grip_mode, sim2real dc76947): the finger closes slowly in
mode 0 until it touches, its angle target is pinned at the measured angle, force_set is set, then it
switches to mode 1 (force closed loop) and holds; then back to mode 0 and open. Prints the switch
latency (request -> /hand_<s>/finger_mode), the held force, and how far the finger moved at each switch.

    source /opt/ros/humble/setup.bash; source ~/rl_ws/robot_control/ros_ws/install/setup.bash
    ROS_DOMAIN_ID=126 python3 scripts/hand_mode_switch_check.py --side right --finger index --force 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, SetForce1, SetSpeed1
from std_msgs.msg import Int32MultiArray

SLOTS = ("pinky", "ring", "middle", "index", "thumb_2", "thumb_1")
OPEN = {"pinky": 1740, "ring": 1740, "middle": 1740, "index": 1740, "thumb_2": 1350}
CLOSED = {"pinky": 900, "ring": 900, "middle": 900, "index": 900, "thumb_2": 1100}
ROOT = Path(__file__).resolve().parents[1]


class Check(Node):
    def __init__(self, side: str) -> None:
        super().__init__("hand_mode_switch_check")
        ns = f"/hand_{side}"
        self.angle = self.force = self.current = self.modes = None
        self.mode_t = None
        self.hand_id = 1
        self.angle_pub = self.create_publisher(SetAngle1, f"{ns}/angle_set", 10)
        self.speed_pub = self.create_publisher(SetSpeed1, f"{ns}/speed_set", 10)
        self.force_pub = self.create_publisher(SetForce1, f"{ns}/force_set", 10)
        self.mode_pub = self.create_publisher(Int32MultiArray, f"{ns}/finger_mode_set", 10)
        self.create_subscription(GetAngleAct1, f"{ns}/angle_actual", self._on_angle, 10)
        self.create_subscription(GetForceAct1, f"{ns}/force_actual",
                                 lambda m: setattr(self, "force", [int(v) for v in m.joint_values]), 10)
        self.create_subscription(GetCurrentAct1, f"{ns}/current_actual",
                                 lambda m: setattr(self, "current", [int(v) for v in m.joint_values]), 10)
        self.create_subscription(Int32MultiArray, f"{ns}/finger_mode", self._on_mode,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _on_angle(self, m) -> None:
        self.angle, self.hand_id = [int(v) for v in m.joint_values], int(m.hand_id)

    def _on_mode(self, m) -> None:
        self.modes, self.mode_t = [int(v) for v in m.data], time.monotonic()

    def send(self, msg_type, pub, values) -> None:
        msg = msg_type()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = self.hand_id
        msg.joint_values = [int(v) for v in values]
        pub.publish(msg)

    def spin_for(self, seconds: float, until=None) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.002)
            if until is not None and until():
                return True
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--finger", choices=SLOTS[:5], default="index")
    ap.add_argument("--force", type=int, default=400, help="force_set while held (g, firmware units)")
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--contact-g", type=float, default=150.0)
    args = ap.parse_args(argv)
    slot = SLOTS.index(args.finger)
    one = lambda reg: [reg if i == slot else -1 for i in range(6)]  # noqa: E731
    rclpy.init()
    p = Check(args.side)
    ok = p.spin_for(5.0, lambda: None not in (p.angle, p.force, p.current, p.modes))
    if not ok:
        print("✗ no angle / force / current / finger_mode from the driver (new driver with finger_mode?)")
        return 1
    if any(p.modes):
        print(f"✗ finger modes {p.modes}: expected all 0 at the start")
        return 1
    rest = p.force[slot]
    log: dict = {"side": args.side, "finger": args.finger, "force_set": args.force, "rest_g": rest}
    print(f"{args.side} {args.finger}: rest {rest} g, modes {p.modes}")
    try:
        # 1. close slowly in position mode until it touches
        p.send(SetSpeed1, p.speed_pub, [2000 if i != slot else 300 for i in range(6)])
        p.send(SetAngle1, p.angle_pub, one(OPEN[args.finger]))
        p.spin_for(1.5)
        p.send(SetAngle1, p.angle_pub, one(CLOSED[args.finger]))
        touched = p.spin_for(10.0, lambda: p.force[slot] - rest >= args.contact_g)
        contact = p.angle[slot]
        p.send(SetAngle1, p.angle_pub, one(contact))  # pin the target where it is
        p.spin_for(0.3)
        log.update(touched=touched, contact_reg=contact, force_at_contact=p.force[slot])
        print(f"1 contact {'yes' if touched else 'NO'} at {contact}, force {p.force[slot]} g, current {p.current[slot]} mA")
        if not touched:
            return 1
        # 2. force_set, then mode 1
        before = p.angle[slot]
        p.send(SetForce1, p.force_pub, [args.force if i == slot else 600 for i in range(6)])
        p.spin_for(0.05)
        t0 = time.monotonic()
        p.mode_pub.publish(Int32MultiArray(data=[1 if i == slot else -1 for i in range(6)]))
        acked = p.spin_for(2.0, lambda: p.modes[slot] == 1)
        ms = (p.mode_t - t0) * 1000 if acked and p.mode_t else None
        trace = []
        p.spin_for(args.hold_s, lambda: trace.append((p.angle[slot], p.force[slot], p.current[slot])) and False)
        held = [f for _, f, _ in trace[len(trace) // 2:]]
        log.update(to_force_ms=ms, held_g=sum(held) / max(len(held), 1), held_min=min(held, default=None),
                   held_max=max(held, default=None), angle_after_switch=trace[-1][0] if trace else None,
                   angle_before_switch=before)
        print(f"2 mode 1 {'ack' if acked else 'NO ACK'} in {ms and round(ms, 1)} ms; held {log['held_g']:.0f} g "
              f"({log['held_min']}..{log['held_max']}), angle {before} -> {log['angle_after_switch']}, "
              f"current {trace[-1][2] if trace else None} mA")
        # 3. pin the angle at the measured one, back to mode 0
        pinned = p.angle[slot]
        p.send(SetAngle1, p.angle_pub, one(pinned))
        p.spin_for(0.05)
        t0 = time.monotonic()
        p.mode_pub.publish(Int32MultiArray(data=[0 if i == slot else -1 for i in range(6)]))
        acked0 = p.spin_for(2.0, lambda: p.modes[slot] == 0)
        ms0 = (p.mode_t - t0) * 1000 if acked0 and p.mode_t else None
        p.spin_for(1.0)
        log.update(to_position_ms=ms0, angle_pinned=pinned, angle_after_back=p.angle[slot],
                   force_after_back=p.force[slot])
        print(f"3 mode 0 {'ack' if acked0 else 'NO ACK'} in {ms0 and round(ms0, 1)} ms; angle {pinned} -> "
              f"{p.angle[slot]}, force {p.force[slot]} g")
    finally:
        p.mode_pub.publish(Int32MultiArray(data=[0] * 6))
        p.spin_for(0.3)
        p.send(SetForce1, p.force_pub, [600] * 6)
        p.send(SetSpeed1, p.speed_pub, [2000] * 6)
        p.send(SetAngle1, p.angle_pub, one(OPEN[args.finger]))
        p.spin_for(1.5)
        print(f"4 opened: angle {p.angle[slot]}, force {p.force[slot]} g, modes {p.modes}")
        out = ROOT / "logs" / "hand" / f"mode_switch_{args.side}_{args.finger}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(log, indent=1))
        print(f"log {out}")
        p.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
