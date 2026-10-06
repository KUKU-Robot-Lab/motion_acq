#!/usr/bin/env python3
"""RH56F1 contact overshoot vs closing speed, one finger against a fixed object (real hand).

10.06 cup grasp: force_set 600 g but force_actual reached 1715 g. Tan, Xie, Correll (arXiv
2603.08988, RH56DFX) show the overshoot past the force threshold grows with the closing speed
and that "fast in free space, slow in contact" removes it. This probe measures it on our hand:
for each speed the finger closes from open onto an object the operator holds still in its way,
and the peak / held force and current are recorded. 10.06 first run (left index): the force
reached 1600-1850 g at every speed that touched (force_set 600), and stayed there with the motor
current back at 0 (the actuator holds the squeeze); so this probe also measures whether force_set
does anything and how the held force grows with how far the target is past the contact.

MOVES THE REAL HAND: run only with the operator's approval, the hand driver up
(rh56f1_driver.py --side <s>, ROS_DOMAIN_ID=126) and nothing else commanding the hand
(console hand [끄기]). Only the chosen finger moves; the others hold (-1).

    source /opt/ros/humble/setup.bash; source ~/rl_ws/robot_control/ros_ws/install/setup.bash
    ROS_DOMAIN_ID=126 python3 scripts/hand_contact_probe.py --side left --finger index   # or thumb_2, thumb_1

Trials: find the contact (slow close from open); depth (target 0..80 registers past the contact,
steady force); force_set (100..1000 g, target fully closed: does the hand stop at it?); speed
(2000 / 500 / 150 from 60 registers before the contact, target fully closed).
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
LEAVE = -1
# open / closed register per slot (closing always lowers the register). thumb_1: away from the palm
# (thumb_1 0.6 rad) .. across it (2.09 rad), hand map 10.05/10.06; thumb_2: straight .. bent.
OPEN = {"pinky": 1740, "ring": 1740, "middle": 1740, "index": 1740, "thumb_2": 1350}
CLOSED = {"pinky": 900, "ring": 900, "middle": 900, "index": 900, "thumb_2": 1100, "thumb_1": 615}
THUMB_1_OPEN = {"left": 1473, "right": 1410}
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
            if sample is not None and sample():
                return


def one_trial(p: Probe, slot: int, speed: int, force: int, hybrid: int | None, close_s: float, rest: float,
              contact_g: float, contact_ma: float, open_reg: int, target: int, start: int | None = None,
              hold_after_contact_s: float | None = None) -> dict:
    """Open, (fast to `start` if given), then close towards `target` at `speed` for close_s."""
    six = lambda v, other=2000: [v if i == slot else other for i in range(6)]  # noqa: E731
    one = lambda reg: [reg if i == slot else LEAVE for i in range(6)]  # noqa: E731
    p.send(SetForce1, p.force_pub, [force] * 6)
    p.send(SetSpeed1, p.speed_pub, six(2000))
    p.send(SetAngle1, p.angle_pub, one(open_reg))
    p.spin_for(1.5)
    if start is not None:
        p.send(SetAngle1, p.angle_pub, one(start))
        p.spin_for(1.0)
    p.send(SetSpeed1, p.speed_pub, six(2000 if hybrid else speed))
    p.spin_for(0.05)
    rows: list[tuple] = []
    switched: list[float | None] = [None]
    t0 = time.monotonic()

    touched: list[float | None] = [None]

    def sample() -> bool:
        if p.angle is None or p.force is None or p.current is None:
            return False
        t = time.monotonic() - t0
        f, i = abs(p.force[slot] - rest), abs(p.current[slot])  # thumb_1 reads negative when loaded (10.06)
        if hybrid and switched[0] is None and (f >= contact_g or i >= contact_ma):
            p.send(SetSpeed1, p.speed_pub, six(hybrid))
            switched[0] = t
        if not rows or rows[-1][0] < t - 0.002:
            rows.append((t, p.angle[slot], p.force[slot], p.current[slot]))
        if touched[0] is None and (f >= contact_g or i >= contact_ma):
            touched[0] = t
        return hold_after_contact_s is not None and touched[0] is not None and t - touched[0] >= hold_after_contact_s

    p.send(SetAngle1, p.angle_pub, one(target))
    p.spin_for(close_s, sample)
    status = p.status or {}
    p.send(SetSpeed1, p.speed_pub, six(2000))
    p.send(SetAngle1, p.angle_pub, one(open_reg))
    p.spin_for(1.5)
    contact = next((a for _, a, f, i in rows if abs(f - rest) >= contact_g or abs(i) >= contact_ma), None)
    peak_f = max((r[2] for r in rows), default=None)
    peak_i = max((abs(r[3]) for r in rows), default=None)
    final = rows[-1] if rows else None
    return {"speed": speed, "hybrid": hybrid, "force_set": force, "rest_force": rest, "target": target,
            "start": start, "contact_reg": contact,
            "peak_force": peak_f, "overshoot_g": None if peak_f is None else peak_f - rest - force,
            "final_force": None if final is None else final[2], "peak_current_ma": peak_i,
            "final_angle": None if final is None else final[1], "switched_at_s": switched[0],
            "status": status.get("status"), "error": status.get("error"), "samples": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--finger", choices=SLOTS, default="index")
    ap.add_argument("--speeds", default="2000,500,150", help="closing speeds from just before contact")
    ap.add_argument("--forces", default="100,300,600,1000", help="force_set values to try (does the hand honour it?)")
    ap.add_argument("--depths", default="0,10,20,40,80", help="target this many registers past the contact")
    ap.add_argument("--force", type=int, default=600)
    ap.add_argument("--slow", type=int, default=300, help="speed of the force / depth trials")
    ap.add_argument("--close-s", type=float, default=2.0)
    ap.add_argument("--contact-g", type=float, default=120.0)
    ap.add_argument("--contact-ma", type=float, default=450.0)
    ap.add_argument("--pre", type=int, default=60, help="registers before the contact where slow trials start")
    args = ap.parse_args(argv)
    slot = SLOTS.index(args.finger)
    open_reg = THUMB_1_OPEN[args.side] if args.finger == "thumb_1" else OPEN[args.finger]
    closed = CLOSED[args.finger]
    rclpy.init()
    p = Probe(args.side)
    p.spin_for(1.0)
    if p.angle is None or p.force is None or p.current is None:
        print("✗ no angle/force/current from the hand driver (ROS_DOMAIN_ID=126? driver up?)")
        return 1
    rests = []
    p.spin_for(0.5, lambda: rests.append(p.force[slot]))
    rest = statistics.median(rests)
    print(f"{args.side} {args.finger}: rest force {rest:.0f} g, open {open_reg}, closed {closed}. Object in its way, still.")
    out = ROOT / "logs" / "hand" / f"contact_probe_{args.side}_{args.finger}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    common = dict(close_s=args.close_s, rest=rest, contact_g=args.contact_g, contact_ma=args.contact_ma, open_reg=open_reg)

    def report(fh, tag: str, r: dict) -> dict:
        fh.write(json.dumps({"trial": tag, **r}) + "\n")
        fh.flush()
        print(f"{tag:<16} peak {r['peak_force']} g, final {r['final_force']} g (rest {rest:.0f}), peak {r['peak_current_ma']} mA, "
              f"contact {r['contact_reg']}, final angle {r['final_angle']}, status {r['status']} error {r['error']}", flush=True)
        return r

    try:
        with out.open("w") as fh:
            # 1. find the contact: slow close from open
            r = report(fh, "find contact", one_trial(p, slot, args.slow, args.force, None, target=closed,
                                                       hold_after_contact_s=0.5,
                                                       **{**common, "close_s": max(args.close_s, 2.5 * 2000 / args.slow)}))
            contact = r["contact_reg"]
            if contact is None:
                print("✗ no contact found: the object is not in the way (or contact_g too high)")
                return 1
            start = min(contact + args.pre, open_reg)
            # 2. depth: steady force vs how far the target is past the contact (grip guard margin)
            for d in [int(x) for x in args.depths.split(",")]:
                report(fh, f"depth {d}", one_trial(p, slot, args.slow, args.force, None, target=max(contact - d, closed),
                                                   start=start, **common))
            # 3. force_set: does the hand stop at it? (target fully closed)
            for f in [int(x) for x in args.forces.split(",")]:
                report(fh, f"force_set {f}", one_trial(p, slot, args.slow, f, None, target=closed, start=start, **common))
            # 4. contact speed (from just before the contact, target fully closed)
            for sp in [int(x) for x in args.speeds.split(",")]:
                report(fh, f"speed {sp}", one_trial(p, slot, sp, args.force, None, target=closed, start=start, **common))
    finally:
        p.send(SetSpeed1, p.speed_pub, [2000] * 6)
        p.send(SetAngle1, p.angle_pub, [open_reg if i == slot else LEAVE for i in range(6)])
        p.spin_for(0.8)
        p.destroy_node()
        rclpy.shutdown()
    print(f"log {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
