#!/usr/bin/env python3
"""Admittance on the real hand, finger by finger (--joints, default all five but thumb_1), scripted operator
target (no glove) — run with approval. Every tested finger is opened first; thumb_1 is never moved.

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
    ROS_DOMAIN_ID=126 python3 scripts/hand_admittance_check.py --side right
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
ABORT_G = 1500.0   # a trial opens the finger past this force over rest
ABORT_MA = 950.0   # ... or this current (driver limit 800 mA)
JOINTS = ("index_1", "middle_1", "ring_1", "pinky_1", "thumb_2")
PENETRATIONS = {"thumb_2": "0.05,0.1,0.2"}   # thumb_2 range ~0.47 rad
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

    def reg_per_rad(self, joint: str) -> float:
        q = {**self.rad(), joint: 0.2}
        a = self.map.to_registers(q, side=self.side)[self.slot(joint)]
        b = self.map.to_registers({**q, joint: 0.3}, side=self.side)[self.slot(joint)]
        return abs(a - b) / 0.1

    def slot(self, joint: str) -> int:
        return next(a.slot for a in self.map.axes_of(self.side) if a.name == joint)

    def spin(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.002)


def ramp_trial(h: Hand, joint: str, q_contact: float, pen: float, speed: float, hold_s: float) -> dict:
    """Operator target: q_contact - 0.2 -> q_contact + pen at speed, hold, back. Admittance on. Stops the trial
    (opens) past ABORT_G or ABORT_MA."""
    h.adm.reset()
    start, goal = q_contact - 0.2, q_contact + pen
    h.command(joint, start)
    h.spin(1.2)
    rows, aborted = [], False
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
            q, off = q_op, h.drv_offset[h.slot(joint)] / h.reg_per_rad(joint)
        else:
            offsets = h.adm.update(now, force, h.guard.tips_now(now))
            q = q_op - offsets.get(joint, 0.0)
            caps = h.guard.ceilings(measured, now)
            for j, top in h.adm.ceilings(now, measured, {joint: rows[-1][2]} if rows else None).items():
                caps[j] = min(top, caps.get(j, top))
            if joint in caps:
                q = min(q, caps[joint])
            reg, off = h.command(joint, q), offsets.get(joint, 0.0)
        cur = h.guard.current[0].get(joint) if h.guard.current else None
        rows.append((round(t, 4), round(q_op, 4), round(q, 4), reg, round(measured[joint], 4),
                     round(force.get(joint, 0.0), 1), off, cur))
        if rows[-1][5] > ABORT_G or (cur is not None and cur > ABORT_MA):
            aborted = True
            break
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
    # where this trial actually touched (the object may have moved since the search): first 150 g
    touched = next((r[4] for r in rows if r[5] > 150.0), None)
    return {"penetration": pen, "speed": speed, "peak_g": max(forces, default=None), "aborted": aborted,
            "trial_contact": touched, "real_penetration": None if touched is None else round(goal - touched, 4),
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


def find_contact(h: Hand, j: str, stiff_stop_g: float) -> tuple[float, float] | None:
    """(first touch, stiff contact) in rad: slow position close to 120 g, then 0.1 rad/s to stiff_stop_g."""
    open_q = h.limits[j][0]
    q, touch = open_q, None
    t_end = time.monotonic() + 8.0
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
        return None
    trace = []
    t_end = time.monotonic() + 4.0
    while time.monotonic() < t_end and q < h.limits[j][1]:
        q += 0.1 / RATE
        h.command(j, q)
        h.spin(1.0 / RATE)
        f = h.guard.relative_force(time.monotonic()).get(j, 0.0)
        trace.append((h.rad()[j], f))
        if f >= stiff_stop_g:
            break
    h.command(j, open_q)
    h.spin(1.0)
    stiff = stiff_contact(trace) if trace else None
    if stiff is None:
        stiff = trace[-1][0] if trace else touch
        print(f"  ! the object kept giving way up to {trace[-1][1] if trace else 0:.0f} g: contact = that point (soft)")
    return touch, stiff


def run_joint(h: Hand, j: str, args, adm: dict, fh) -> list[dict]:
    found = find_contact(h, j, args.stiff_stop_g)
    if found is None:
        print(f"✗ {j}: no contact found, the object is not in its way")
        return []
    touch, contact = found
    rpr = h.reg_per_rad(j)
    k_rad = adm["k_g_per_reg"] * rpr if args.via == "driver" else h.adm.config.stiffness_g_per_rad
    print(f"{j}: first touch {touch:.3f} rad, stiff contact {contact:.3f} rad (+{(contact - touch) * rpr:.0f} registers)")
    pens = args.penetrations or PENETRATIONS.get(j, "0.1,0.2,0.4")
    out = []
    for sp in [float(v) for v in args.speeds.split(",")]:
        for pen in [float(v) for v in pens.split(",")]:
            r = ramp_trial(h, j, contact, pen, sp, args.hold_s)
            real = r["real_penetration"] if r["real_penetration"] is not None else pen
            r["expected_g"] = round(min(k_rad * max(real, 0.0), adm["f_max_g"]), 1)
            r["pass"] = (not r["aborted"]) and trial_pass(r, adm["hold_band_g"])
            fh.write(json.dumps({"joint": j, "touch": touch, "contact": contact, **r}) + "\n")
            fh.flush()
            print(f"  {'✓' if r['pass'] else '✗'} pen {pen:.2f} (real {real:.3f}) rad @ {sp:.1f} rad/s: held {r['held_g']} g "
                  f"/ expected {r['expected_g']} g (range {r['held_range_g']}), peak {r['peak_g']} g / {r['peak_ma']} mA"
                  f"{'  ABORTED' if r['aborted'] else ''}", flush=True)
            out.append(r)
            if r["aborted"]:
                print(f"  {j}: stopped after an abort (force > {ABORT_G:.0f} g or current > {ABORT_MA:.0f} mA)")
                return out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--side", choices=("left", "right"), required=True)
    ap.add_argument("--joints", default=",".join(JOINTS), help=f"comma list from {JOINTS}, run one after the other")
    ap.add_argument("--penetrations", default=None, help="rad past the contact (default 0.1,0.2,0.4; thumb_2 0.05,0.1,0.2)")
    ap.add_argument("--speeds", default="0.5,4.0", help="operator target speeds, rad/s")
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--stiff-stop-g", type=float, default=350.0,
                    help="contact search: close past the first touch until this force over rest")
    ap.add_argument("--via", choices=("driver", "local"), default="driver",
                    help="driver: angle_target (the driver's 500 Hz admittance); local: this script at 120 Hz")
    args = ap.parse_args(argv)
    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    bad = [j for j in joints if j not in JOINTS]
    if bad:
        ap.error(f"unknown joints {bad}")
    rclpy.init()
    h = Hand(args.side, args.via)
    for _ in range(50):
        h.spin(0.1)
        if h.reg is not None and h.guard.force is not None:
            break
    if h.reg is None or h.guard.force is None:
        print("✗ no angle / force from the driver")
        return 1
    h.send(SetSpeed1, h.speed_pub, [2000] * 6)
    h.send(SetForce1, h.force_pub, [600] * 6)
    for j in JOINTS:   # every tested finger open (thumb_1 left where it is), then the rest bias
        h.command(j, h.limits[j][0])
        h.spin(0.05)
    h.spin(1.5)
    for _ in range(60):
        h.spin(1.0 / RATE)
        h.guard.learn_rest(time.monotonic())
    print(f"{args.side}: rest bias {h.guard.bias} g; thumb_1 {h.rad()['thumb_1']:.3f} rad (not moved)")
    adm = contract_admittance()
    out = ROOT / "logs" / "hand" / f"admittance_check_{args.side}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    try:
        with out.open("w") as fh:
            for j in joints:
                summary[j] = run_joint(h, j, args, adm, fh)
                h.command(j, h.limits[j][0])
                h.spin(0.5)
    finally:
        for j in JOINTS:
            h.command(j, h.limits[j][0])
            h.spin(0.05)
        h.spin(1.0)
        h.destroy_node()
        rclpy.shutdown()
    print("\nsummary (held / expected g, worst hold range, peak g / mA):")
    for j, rs in summary.items():
        if not rs:
            print(f"  {j:9s} no contact")
            continue
        ok = sum(r["pass"] for r in rs)
        pairs = " ".join(f"{r['held_g']:.0f}/{r['expected_g']:.0f}" for r in rs if r["held_g"] is not None)
        print(f"  {j:9s} {ok}/{len(rs)} pass | {pairs} | range <= {max(r['held_range_g'] or 0 for r in rs):.0f} | "
              f"peak {max(r['peak_g'] or 0 for r in rs):.0f} g / {max(r['peak_ma'] or 0 for r in rs):.0f} mA")
    print(f"log {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
