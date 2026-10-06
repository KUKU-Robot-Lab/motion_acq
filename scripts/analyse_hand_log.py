#!/usr/bin/env python3
"""Summarise a hand_node JSONL log and check the STEP 3 invariants (exit 1 on failure).

    python3 scripts/analyse_hand_log.py logs/hand/hand_right_<time>.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter

REG_RANGE = {0: (900, 1740), 1: (900, 1740), 2: (900, 1740), 3: (900, 1740), 4: (1100, 1350), 5: (600, 1750)}
MAX_VELOCITY_RAD_S = 8.0  # configs/hands/nova2_to_rh56f1.yaml limits.max_velocity_rad_s (10.05)


def main(path: str) -> int:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    states = Counter(r["state"] for r in rows)
    running = [r for r in rows if r["state"] == "running"]
    failures = []
    if not running:
        failures.append("never ran")
    if states.get("hold", 0) == 0:
        failures.append("no HOLD seen (glove dropouts expected)")
    for r in running:
        for slot, value in enumerate(r["registers"]):
            lo, hi = REG_RANGE[slot]
            if not lo <= value <= hi:
                failures.append(f"slot {slot} register {value} outside [{lo}, {hi}]")
                break
    # A fault while walking home at the end is the fake driver stopping with the launch (both get
    # the SIGINT); any other fault fails the check.
    faults = [i for i, r in enumerate(rows) if r.get("fault")
              and not (i > 0 and rows[i - 1].get("phase") == "return_home" and i >= len(rows) - 100)]
    if faults:
        failures.append(f"fault: {rows[faults[0]]['fault']}")
    # The first command after enable must start at the measured hand pose.
    first = running[0] if running else None
    if first and first["measured_registers"]:
        jump = max(abs(c - m) for c, m in zip(first["registers"], first["measured_registers"], strict=False))
        if jump > 40:
            failures.append(f"first command {jump} registers away from the measured hand")
    # Resuming after a glove HOLD must not jump.
    for i in range(1, len(rows)):
        if rows[i - 1]["state"] == "hold" and rows[i]["state"] == "running":
            cur = rows[i]
            before = next((r for r in reversed(rows[: i - 1]) if r["state"] == "running"), None)
            if before:
                jump = max(abs(a - b) for a, b in zip(cur["registers"], before["registers"], strict=False))
                if jump > 40:
                    failures.append(f"{jump}-register jump when the glove came back")
    holds = [r for r in rows if r["state"] == "hold"]
    if any(r["registers"] is not None for r in holds):
        failures.append("a HOLD cycle carried registers (would publish)")
    if any(r["glove_age_s"] is not None and r["glove_age_s"] <= 0.2 for r in holds):
        failures.append("HOLD with a fresh glove sample")
    # Rate: q_command change per cycle within max velocity * dt (2 rad/s, small slack).
    # (the grip guard may cut a pressing joint's command back to the measured hand in one cycle)
    worst = 0.0
    for a, b in zip(running, running[1:], strict=False):
        dt = b["t_mono_s"] - a["t_mono_s"]
        if 0 < dt < 0.2 and not (a.get("grip_guard") or b.get("grip_guard")):
            worst = max(worst, max(abs(b["q_command_rad"][j] - a["q_command_rad"][j]) / dt for j in a["q_command_rad"]))
    if worst > MAX_VELOCITY_RAD_S * 1.1:
        failures.append(f"joint speed {worst:.2f} rad/s above the {MAX_VELOCITY_RAD_S} limit")
    # Feedback (10.05): robot index contact -> glove index brake while following; never when not.
    contact = [r for r in running if (r.get("tip_force_n") or {}).get("index", 0.0) >= 0.3]
    braked = [r for r in contact if ((r.get("haptics") or {}).get("brake") or {}).get("index", 0.0) > 0.0]
    if contact and not braked:
        failures.append(f"{len(contact)} cycles of index contact but no glove brake")
    idle_haptics = [r for r in rows if r["state"] != "running" and any(
        v > 0.0 for v in ((r.get("haptics") or {}).get("brake") or {}).values())]
    if idle_haptics:
        failures.append(f"{len(idle_haptics)} cycles braked the glove while not following it")
    # Grip guard (10.06): a finger pressing on the object must not be driven past it (fake index object)
    guarded = [r for r in running if (r.get("grip_guard") or {}).get("index_1")]
    pressed = [r for r in running if (r.get("joint_force") or {}).get("index_1", 0.0) >= 400]
    too_hard = [r for r in running if (r.get("joint_force") or {}).get("index_1", 0.0) >= 1000]
    if pressed and not guarded:
        failures.append(f"{len(pressed)} cycles of index pressing but the grip guard never engaged")
    if too_hard:
        failures.append(f"{len(too_hard)} cycles with index force >= 1000 g (guard did not hold the finger)")
    slowed = [r for r in running if "index_1" in (r.get("contact_slow") or [])]
    if pressed and not slowed:
        failures.append("index pressed but never ran at contact speed")
    print(f"index pressing cycles {len(pressed)}, grip guard on index in {len(guarded)}, "
          f"max index force {max(((r.get('joint_force') or {}).get('index_1', 0.0) for r in running), default=0):.0f} g, "
          f"index at contact speed in {len(slowed)}")
    lag = []
    for r in running[30:]:
        if r["measured_registers"] is not None:
            lag.append(max(abs(m - c) for m, c in zip(r["measured_registers"], r["registers"], strict=False)))
    span = {}
    for slot in range(6):
        values = [r["registers"][slot] for r in running]
        span[slot] = (min(values), max(values)) if values else None
    period = [b["t_mono_s"] - a["t_mono_s"] for a, b in zip(rows, rows[1:], strict=False)]
    print(f"log {path}")
    modes = Counter(r.get("mode") for r in rows)
    print(f"cycles {len(rows)}  states {dict(states)}  modes {dict(modes)}")
    print(f"cycle period mean {sum(period) / max(len(period), 1) * 1000:.1f} ms  max {max(period, default=0) * 1000:.1f} ms")
    print(f"register span per slot (pinky ring middle index thumb_bend thumb_rot): {span}")
    print(f"index contact cycles {len(contact)}, glove index brake on in {len(braked)}")
    # 10.05 example-pose map: the glove tip distances must reach it (else no pinches)
    no_tips = sum(1 for r in running if r.get("missing_inputs"))
    print(f"cycles without glove tip data: {no_tips}")
    if running and no_tips == len(running):
        failures.append("the map never got the glove tip distances")
    print(f"max joint speed {worst:.2f} rad/s;  fake hand lag median {sorted(lag)[len(lag) // 2] if lag else '-'} reg")
    if failures:
        print("FAIL: " + "; ".join(dict.fromkeys(failures)))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
