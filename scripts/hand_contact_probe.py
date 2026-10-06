#!/usr/bin/env python3
"""RH56F1 contact overshoot vs closing speed, one finger against a fixed object (real hand).

10.06 cup grasp: force_set 600 g but force_actual reached 1715 g. Tan, Xie, Correll (arXiv
2603.08988, RH56DFX) show the overshoot past the force threshold grows with the closing speed
and that "fast in free space, slow in contact" removes it. This probe measures it on our hand:
for each speed the finger closes from open onto an object the operator holds still in its way,
and the peak force / current over force_set is recorded. --hybrid runs the motion_acq rule
(2000 until the first sign of contact, then the contact speed) to check that a speed change
mid-motion takes effect.

MOVES THE REAL HAND: run only with the operator's approval, the hand driver up
(rh56f1_driver.py --side <s>, ROS_DOMAIN_ID=126) and nothing else commanding the hand
(console hand [끄기]). Only the chosen finger moves; the others hold (-1).

    source /opt/ros/humble/setup.bash; source ~/rl_ws/robot_control/ros_ws/install/setup.bash
    ROS_DOMAIN_ID=126 python3 scripts/hand_contact_probe.py --side left --finger index \\
        --speeds 2000,1000,500,300,150 --repeats 2 [--hybrid 300]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rh56f1_interfaces.msg import GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, SetForce1, SetSpeed1
from std_msgs.msg import String

SLOTS = ("pinky", "ring", "middle", "index", "thumb_2", "thumb_1")  # driver slot order
OPEN_REG = 1740
CLOSED_REG = 900
LEAVE = -1
ROOT = Path(__file__).resolve().parents[1]


class Probe(Node):
    def __init__(self, side: str) -> None:
        super().__init__("hand_contact_probe")
        ns = f"/hand_{side}"
        self.angle_pub = self.create_publisher(SetAngle1, f"{ns}/angle_set", 10)
        self.speed_pub = self.create_publisher(SetSpeed1, f"{ns}/speed_set", 10)
        self.force_pub = self.create_publisher(SetForce1, f"{ns}/force_set", 10)
        self.angle = self.force = self.current = None
        self.status: dict | None = None
        self.hand_id = 1
        self.create_subscription(GetAngleAct1, f"{ns}/angle_actual", self._on_angle, 10)
        self.create_subscription(GetForceAct1, f"{ns}/force_actual", lambda m: setattr(self, "force", [int(v) for v in m.joint_values]), 10)
        self.create_subscription(GetCurrentAct1, f"{ns}/current_actual", lambda m: setattr(self, "current", [int(v) for v in m.joint_values]), 10)
        self.create_subscription(String, f"{ns}/ecat_status", self._on_status, 10)

    def _on_angle(self, m: GetAngleAct1) -> None:
        self.angle, self.hand_id = [int(v) for v in m.joint_values], int(m.hand_id)

    def _on_status(self, m: String) -> None:
        try:
            self.status = json.loads(m.data)
        except ValueError:
            pass

    def send(self, msg_type, pub, values) -> None:
        msg = msg_type()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = self.hand_id
        msg.joint_values = [int(v) for v in values]
        pub.publish(msg)

    def spin_for(self, seconds: float, sample=None) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.002)
            if sample is not None:
                sample()


def one_trial(p: Probe, slot: int, speed: int, force: int, hybrid: int | None, close_s: float, rest: float,
              contact_g: float, contact_ma: float) -> dict:
    six = lambda v, other=2000: [v if i == slot else other for i in range(6)]  # noqa: E731
    p.send(SetForce1, p.force_pub, [force] * 6)
    p.send(SetSpeed1, p.speed_pub, six(2000))
    p.send(SetAngle1, p.angle_pub, [OPEN_REG if i == slot else LEAVE for i in range(6)])
    p.spin_for(1.5)
    p.send(SetSpeed1, p.speed_pub, six(2000 if hybrid else speed))
    p.spin_for(0.05)
    rows: list[tuple] = []
    switched: list[float | None] = [None]
    t0 = time.monotonic()

    def sample() -> None:
        if p.angle is None or p.force is None or p.current is None:
            return
        t = time.monotonic() - t0
        f, i = p.force[slot] - rest, abs(p.current[slot])
        if hybrid and switched[0] is None and (f >= contact_g or i >= contact_ma):
            p.send(SetSpeed1, p.speed_pub, six(hybrid))
            switched[0] = t
        if not rows or rows[-1][0] < t - 0.002:
            rows.append((t, p.angle[slot], p.force[slot], p.current[slot]))

    p.send(SetAngle1, p.angle_pub, [CLOSED_REG if i == slot else LEAVE for i in range(6)])
    p.spin_for(close_s, sample)
    status = p.status or {}
    p.send(SetSpeed1, p.speed_pub, six(2000))
    p.send(SetAngle1, p.angle_pub, [OPEN_REG if i == slot else LEAVE for i in range(6)])
    p.spin_for(1.5)
    peak_f = max((r[2] for r in rows), default=None)
    peak_i = max((abs(r[3]) for r in rows), default=None)
    final = rows[-1] if rows else None
    return {"speed": speed, "hybrid": hybrid, "force_set": force, "rest_force": rest,
            "peak_force": peak_f, "overshoot_g": None if peak_f is None else peak_f - rest - force,
            "final_force": None if final is None else final[2], "peak_current_ma": peak_i,
            "final_angle": None if final is None else final[1], "switched_at_s": switched[0],
            "status": status.get("status"), "error": status.get("error"), "samples": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--finger", choices=SLOTS[:4], default="index")
    ap.add_argument("--speeds", default="2000,1000,500,300,150")
    ap.add_argument("--force", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--hybrid", type=int, default=None, help="also run 2000 -> this speed at the first contact sign")
    ap.add_argument("--close-s", type=float, default=2.0)
    ap.add_argument("--contact-g", type=float, default=120.0)
    ap.add_argument("--contact-ma", type=float, default=450.0)
    args = ap.parse_args(argv)
    slot = SLOTS.index(args.finger)
    speeds = [int(s) for s in args.speeds.split(",")]
    rclpy.init()
    p = Probe(args.side)
    p.spin_for(1.0)
    if p.angle is None or p.force is None or p.current is None:
        print("✗ no angle/force/current from the hand driver (ROS_DOMAIN_ID=126? driver up?)")
        return 1
    rests = []
    p.spin_for(0.5, lambda: rests.append(p.force[slot]))
    rest = statistics.median(rests)
    print(f"{args.side} {args.finger}: rest force {rest:.0f} g, force_set {args.force} g. Object in the finger's way, still.")
    plan = [(s, None) for s in speeds for _ in range(args.repeats)]
    if args.hybrid:
        plan += [(2000, args.hybrid)] * args.repeats
    results = []
    out = ROOT / "logs" / "hand" / f"contact_probe_{args.side}_{args.finger}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w") as fh:
            for speed, hybrid in plan:
                r = one_trial(p, slot, speed, args.force, hybrid, args.close_s, rest, args.contact_g, args.contact_ma)
                fh.write(json.dumps(r) + "\n")
                results.append(r)
                tag = f"2000->{hybrid}" if hybrid else str(speed)
                print(f"speed {tag:>9}: peak {r['peak_force']} g (over set {r['overshoot_g']}), "
                      f"final {r['final_force']} g, peak {r['peak_current_ma']} mA, angle {r['final_angle']}, "
                      f"switch {r['switched_at_s']}, status {r['status']} error {r['error']}")
    finally:
        p.send(SetSpeed1, p.speed_pub, [2000] * 6)
        p.send(SetAngle1, p.angle_pub, [OPEN_REG if i == slot else LEAVE for i in range(6)])
        p.spin_for(0.5)
        p.destroy_node()
        rclpy.shutdown()
    print(f"log {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
