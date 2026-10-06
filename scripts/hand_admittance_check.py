#!/usr/bin/env python3
"""Admittance on the real hand, one finger, scripted operator target (no glove) — run with approval.

--via driver (default): sends the target on /hand_<s>/angle_target, the driver's canonical admittance at
500 Hz (robot_control components/rh56f1.yaml) does the rest; offsets read from /hand_<s>/admittance_offset.
--via local: runs motion_acq.hand.admittance (+ the grip guard) here at 120 Hz on angle_set (10.06 first try).
One joint against an object held still in its way. The contact is where the object stops giving way
(10.06: a cup first touched at 120 g then gave 30-60 registers at 5-7 g/register): a slow close past the first
touch up to --stiff-stop-g, the stiff contact from motion_acq.hand.admittance.stiff_contact. The "operator"
target ramps from 0.2 rad before it to `penetration` past it at `speed`, holds, then opens again. Reports per
trial the held force against the expected stiffness x penetration (below 800 g; the contract's k, 10.06 user
(A): the actuator reading, wherever it touches), the peak at impact, oscillation (force range over the last
second of the hold) and the command -> force delay. Pass per trial: held within 25 % (or the hold band) of
expected, range < 100 g, peak < 1200 g, current < 800 mA.

    source /opt/ros/humble/setup.bash; source ~/rl_ws/robot_control/ros_ws/install/setup.bash
    ROS_DOMAIN_ID=126 python3 scripts/hand_admittance_check.py --side right --joint index_1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Int32MultiArray  # noqa: E402
from rh56f1_interfaces.msg import (GetAngleAct1, GetCurrentAct1, GetForceAct1, SetAngle1, SetForce1,  # noqa: E402
                                   SetSpeed1, TouchData1)

import yaml  # noqa: E402

from motion_acq.hand.admittance import Admittance, stiff_contact  # noqa: E402
from motion_acq.hand.feedback import tip_forces_n  # noqa: E402
from motion_acq.hand.grip_guard import GripGuard  # noqa: E402
from motion_acq.hand.retarget import load_hand_retarget_config  # noqa: E402
from motion_acq.hand.rh56f1 import load_rh56f1_map  # noqa: E402

RATE = 120.0
REG_PER_RAD = 550.0  # fingers 1740 open .. 900 closed
CONTRACT = ROOT.parent / "robot_control" / "components" / "rh56f1.yaml"


def contract_admittance() -> dict:
    return yaml.safe_load(CONTRACT.read_text())["control"]["admittance"]


class Hand(Node):
    def __init__(self, side: str, via: str = "driver") -> None:
        super().__init__("hand_admittance_check")
        self.via = via
        self.drv_offset = [0] * 6
        ns = f"/hand_{side}"
        self.side = side
        self.reg = None
        self.hand_id = 1
        cfg = load_hand_retarget_config()
        self.guard = GripGuard(cfg.grip_guard)
        self.adm = Admittance(cfg.admittance)
        self.limits = cfg.limits_rad
        self.map = load_rh56f1_map()
        self.angle_pub = self.create_publisher(SetAngle1, f"{ns}/angle_set", 10)
        self.target_pub = self.create_publisher(SetAngle1, f"{ns}/angle_target", 10)
        self.create_subscription(Int32MultiArray, f"{ns}/admittance_offset",
                                 lambda m: setattr(self, "drv_offset", [int(v) for v in m.data]), 10)
        self.speed_pub = self.create_publisher(SetSpeed1, f"{ns}/speed_set", 10)
        self.force_pub = self.create_publisher(SetForce1, f"{ns}/force_set", 10)
        self.create_subscription(GetAngleAct1, f"{ns}/angle_actual", self._on_angle, 10)
        self.create_subscription(GetForceAct1, f"{ns}/force_actual",
                                 lambda m: self.guard.on_force(list(m.joint_names), list(m.joint_values),
                                                               time.monotonic()), 10)
        self.create_subscription(GetCurrentAct1, f"{ns}/current_actual",
                                 lambda m: self.guard.on_current(list(m.joint_names), list(m.joint_values),
                                                                 time.monotonic()), 10)
        self.create_subscription(TouchData1, f"{ns}/touch_data",
                                 lambda m: self.guard.on_tips(tip_forces_n(list(m.finger_forces)),
                                                              time.monotonic()), 10)

    def _on_angle(self, m) -> None:
        self.reg, self.hand_id = [int(v) for v in m.joint_values], int(m.hand_id)

    def rad(self) -> dict[str, float]:
        return self.map.to_rad(self.reg, side=self.side)

    def send(self, msg_type, pub, values) -> None:
        msg = msg_type()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hand_id = self.hand_id
        msg.joint_values = [int(v) for v in values]
        pub.publish(msg)

    def command(self, joint: str, q: float, target: bool = False) -> int:
        """target: on angle_target (the driver's admittance) instead of angle_set (position)."""
        q = min(max(q, self.limits[joint][0]), self.limits[joint][1])
        regs = self.map.to_registers({**self.rad(), joint: q}, side=self.side)
        slot = self.slot(joint)
        self.send(SetAngle1, self.target_pub if target else self.angle_pub, [regs[i] if i == slot else -1 for i in range(6)])
        return regs[slot]

    def slot(self, joint: str) -> int:
        return next(a.slot for a in self.map.axes_of(self.side) if a.name == joint)

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.002)


def ramp_trial(h: Hand, joint: str, q_contact: float, pen: float, speed: float, hold_s: float) -> dict:
    """Operator target: q_contact - 0.2 -> q_contact + pen at speed, hold, back. Admittance on."""
    h.adm.reset()
    start, goal = q_contact - 0.2, q_contact + pen
    h.command(joint, start)
    h.spin(1.2)
    rows = []
    t0 = time.monotonic()
    next_t = t0
    ramp_s = (goal - start) / speed
    total = ramp_s + hold_s
    while True:
        now = time.monotonic()
        t = now - t0
        if t > total:
            break
        q_op = start + min(t, ramp_s) * speed
        force = h.guard.relative_force(now)
        measured = h.rad()
        if h.via == "driver":   # the driver's admittance: send the operator target, read its offset (registers)
            reg = h.command(joint, q_op, target=True)
            q, off = q_op, h.drv_offset[h.slot(joint)] / 550.0
        else:
            offsets = h.adm.update(now, force, h.guard.tips_now(now))
            q = q_op - offsets.get(joint, 0.0)
            caps = h.guard.ceilings(measured, now)
            for j, top in h.adm.ceilings(measured).items():
                caps[j] = min(top, caps.get(j, top))
            if joint in caps:
                q = min(q, caps[joint])
            reg, off = h.command(joint, q), offsets.get(joint, 0.0)
        cur = h.guard.current[0].get(joint) if h.guard.current else None
        rows.append((round(t, 4), round(q_op, 4), round(q, 4), reg, round(measured[joint], 4),
                     round(force.get(joint, 0.0), 1), off, cur))
        next_t += 1.0 / RATE
        while time.monotonic() < next_t:
            rclpy.spin_once(h, timeout_sec=max(0.0, next_t - time.monotonic()))
    h.command(joint, start)
    h.spin(1.0)
    hold = [r for r in rows if r[0] >= total - 1.0]
    forces = [r[5] for r in rows]
    held = [r[5] for r in hold]
    # delay: first tick the command moves past the contact -> first tick the force rises 60 g
    t_cmd = next((r[0] for r in rows if r[1] > q_contact + 0.01), None)
    t_f = next((r[0] for r in rows if r[5] > 60.0), None)
    currents = [r[7] for r in rows if r[7] is not None]
    return {"penetration": pen, "speed": speed, "peak_g": max(forces, default=None),
            "peak_ma": max(currents, default=None),
            "held_g": statistics.median(held) if held else None,
            "held_range_g": (max(held) - min(held)) if held else None,
            "offset_rad": hold[-1][6] if hold else None, "cmd_to_force_s": None if None in (t_cmd, t_f) else t_f - t_cmd,
            "rows": rows}


def trial_pass(r: dict, hold_band_g: float) -> bool:
    held, exp = r["held_g"], r["expected_g"]
    return (held is not None and abs(held - exp) <= max(0.25 * exp, hold_band_g)
            and r["held_range_g"] < 100.0 and r["peak_g"] < 1200.0
            and (r["peak_ma"] is None or r["peak_ma"] < 800.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--joint", choices=("index_1", "middle_1", "ring_1", "pinky_1", "thumb_2"), default="index_1")
    ap.add_argument("--penetrations", default="0.1,0.2,0.4")
    ap.add_argument("--speeds", default="0.5,4.0", help="operator target speeds, rad/s")
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--stiff-stop-g", type=float, default=350.0,
                    help="contact search: close past the first touch until this force over rest")
    ap.add_argument("--via", choices=("driver", "local"), default="driver",
                    help="driver: angle_target (the driver's 500 Hz admittance); local: this script at 120 Hz")
    args = ap.parse_args(argv)
    rclpy.init()
    h = Hand(args.side, args.via)
    for _ in range(50):
        h.spin(0.1)
        if h.reg is not None and h.guard.force is not None:
            break
    if h.reg is None or h.guard.force is None:
        print("✗ no angle / force from the driver")
        return 1
    j = args.joint
    h.send(SetSpeed1, h.speed_pub, [2000] * 6)
    h.send(SetForce1, h.force_pub, [600] * 6)
    open_q = h.limits[j][0]
    h.command(j, open_q)
    h.spin(1.5)
    for _ in range(60):  # rest bias, open and still
        h.spin(1.0 / RATE)
        h.guard.learn_rest(time.monotonic())
    print(f"{args.side} {j}: rest bias {h.guard.bias.get(j)} g")
    # first touch: slow position close (0.4 rad/s), 120 g over rest
    q = open_q
    t_end = time.monotonic() + 8.0
    touch = None
    while time.monotonic() < t_end and q < h.limits[j][1]:
        q += 0.4 / RATE
        h.command(j, q)
        h.spin(1.0 / RATE)
        if h.guard.relative_force(time.monotonic()).get(j, 0.0) >= 120.0:
            touch = h.rad()[j]
            break
    if touch is None:
        h.command(j, open_q)
        h.spin(1.0)
        print("✗ no contact found: the object is not in the way")
        return 1
    # past it at 0.1 rad/s until --stiff-stop-g (or 4 s): where the object stops giving way
    trace = []
    t_end = time.monotonic() + 4.0
    while time.monotonic() < t_end and q < h.limits[j][1]:
        q += 0.1 / RATE
        h.command(j, q)
        h.spin(1.0 / RATE)
        f = h.guard.relative_force(time.monotonic()).get(j, 0.0)
        trace.append((h.rad()[j], f))
        if f >= args.stiff_stop_g:
            break
    h.command(j, open_q)
    h.spin(1.0)
    stiff = stiff_contact(trace)
    if stiff is None:
        stiff = trace[-1][0]
        print(f"! the object kept giving way up to {trace[-1][1]:.0f} g: contact = that point (soft object)")
    contact = stiff
    print(f"first touch {touch:.3f} rad, stiff contact {contact:.3f} rad "
          f"(+{(contact - touch) * REG_PER_RAD:.0f} registers)")
    adm = contract_admittance()
    k_rad = adm["k_g_per_reg"] * REG_PER_RAD if args.via == "driver" else h.adm.config.stiffness_g_per_rad
    out = ROOT / "logs" / "hand" / f"admittance_check_{args.side}_{j}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w") as fh:
            for sp in [float(v) for v in args.speeds.split(",")]:
                for pen in [float(v) for v in args.penetrations.split(",")]:
                    r = ramp_trial(h, j, contact, pen, sp, args.hold_s)
                    r["expected_g"] = round(min(k_rad * pen, adm["f_max_g"]), 1)
                    r["pass"] = trial_pass(r, adm["hold_band_g"])
                    fh.write(json.dumps({"joint": j, "touch": touch, "contact": contact, **r}) + "\n")
                    fh.flush()
                    print(f"{'✓' if r['pass'] else '✗'} pen {pen:.2f} rad @ {sp:.1f} rad/s: held {r['held_g']} g "
                          f"/ expected {r['expected_g']} g (range {r['held_range_g']}), "
                          f"peak {r['peak_g']} g / {r['peak_ma']} mA, offset {r['offset_rad'] and round(r['offset_rad'], 3)} rad, "
                          f"cmd->force {r['cmd_to_force_s'] and round(r['cmd_to_force_s'] * 1000)} ms", flush=True)
    finally:
        h.command(j, open_q)
        h.spin(1.0)
        h.destroy_node()
        rclpy.shutdown()
    print(f"log {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
