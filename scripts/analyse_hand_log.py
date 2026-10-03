#!/usr/bin/env python3
"""Summarise a hand_node JSONL log and check the STEP 3 invariants (exit 1 on failure).

    python3 scripts/analyse_hand_log.py logs/hand/hand_right_<time>.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter

REG_RANGE = {0: (900, 1740), 1: (900, 1740), 2: (900, 1740), 3: (900, 1740), 4: (1100, 1350), 5: (600, 1750)}


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
    holds = [r for r in rows if r["state"] == "hold"]
    if any(r["registers"] is not None for r in holds):
        failures.append("a HOLD cycle carried registers (would publish)")
    if any(r["glove_age_s"] is not None and r["glove_age_s"] <= 0.2 for r in holds):
        failures.append("HOLD with a fresh glove sample")
    # Rate: q_command change per cycle within max velocity * dt (2 rad/s, 30 Hz, small slack).
    worst = 0.0
    for a, b in zip(running, running[1:]):
        dt = b["t_mono_s"] - a["t_mono_s"]
        if 0 < dt < 0.2:
            worst = max(worst, max(abs(b["q_command_rad"][j] - a["q_command_rad"][j]) / dt for j in a["q_command_rad"]))
    if worst > 2.0 * 1.1:
        failures.append(f"joint speed {worst:.2f} rad/s above the 2.0 limit")
    lag = []
    for r in running[30:]:
        if r["measured_registers"] is not None:
            lag.append(max(abs(m - c) for m, c in zip(r["measured_registers"], r["registers"])))
    span = {}
    for slot in range(6):
        values = [r["registers"][slot] for r in running]
        span[slot] = (min(values), max(values)) if values else None
    period = [b["t_mono_s"] - a["t_mono_s"] for a, b in zip(rows, rows[1:])]
    print(f"log {path}")
    print(f"cycles {len(rows)}  states {dict(states)}  enabled {any(r['enabled'] for r in rows)}")
    print(f"cycle period mean {sum(period) / max(len(period), 1) * 1000:.1f} ms  max {max(period, default=0) * 1000:.1f} ms")
    print(f"register span per slot (pinky ring middle index thumb_bend thumb_rot): {span}")
    print(f"max joint speed {worst:.2f} rad/s;  fake hand lag median {sorted(lag)[len(lag) // 2] if lag else '-'} reg")
    if failures:
        print("FAIL: " + "; ".join(dict.fromkeys(failures)))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
