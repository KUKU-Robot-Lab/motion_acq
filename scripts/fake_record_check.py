#!/usr/bin/env python3
"""Fake end-to-end recording (no hardware): arms + head + both hands -> one LeRobot dataset.

    .venv/bin/python scripts/fake_record_check.py [--station arm4090] [--seconds 8]

Runs `macq station --fake` (mock Quest, fake head bus, fake hand chains on
ROS_DOMAIN_ID=177 localhost-only, teleop-record --fake-robot) in a
pseudo-terminal, presses Space once to start, lets --episode-time-s save the
episode, and checks the dataset. The station stops its own processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin"
LOGS = ROOT / "logs" / "fake_record"
IDENTITY_TCP = """calibration:
  frame_convention: pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion
  controller_to_gripper_tcp:
    left: {position: [0.0, 0.0, 0.0], quaternion: [0.0, 0.0, 0.0, 1.0]}
    right: {position: [0.0, 0.0, 0.0], quaternion: [0.0, 0.0, 0.0, 1.0]}
"""
def wait_for(path: Path, text: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text(errors="replace"):
            return True
        time.sleep(0.2)
    return False


def check_dataset(out: Path, seconds: float, fps: int, arms_only: bool = False) -> list[str]:
    import pandas as pd

    files = sorted(out.glob("data/**/*.parquet"))
    if not files:
        return ["no parquet data written"]
    df = pd.concat(pd.read_parquet(f) for f in files)
    info = json.loads((out / "meta" / "info.json").read_text())
    failures = []
    expected = int(seconds * fps)
    if not 0.8 * expected <= len(df) <= 1.2 * expected:
        failures.append(f"{len(df)} frames, expected ~{expected}")
    needed = ["observation.state", "action"]
    if not arms_only:
        needed += ["observation.head.state", "action.head",
                   "observation.hand.right.state", "action.hand.right", "observation.glove.right.angles",
                   "observation.hand.left.state", "action.hand.left", "observation.glove.left.angles"]
    missing = [k for k in needed if k not in df.columns]
    if missing:
        failures.append(f"missing columns {missing}")
        return failures
    arm = df["observation.state"].map(list).tolist()
    spread = max(max(col) - min(col) for col in zip(*arm, strict=True))
    print(f"  arm observation spread (max over joints) {spread:.3f}")
    if spread < 1e-3:
        failures.append("arm observation never moved")
    if arms_only:
        extra = [c for c in df.columns if c.startswith(("observation.head", "observation.hand", "action.h"))]
        if extra:
            failures.append(f"arms-only dataset has sidecar columns {extra}")
        print(f"  frames {len(df)}")
        return failures
    for key in ("observation.head.status", "observation.hand.right.status", "observation.hand.left.status"):
        values = df[key].map(lambda v: int(v[0]) if hasattr(v, "__len__") else int(v))
        share = (values == 1).mean()
        print(f"  {key}: running share {share:.2f}, values {sorted(values.unique())}")
        if (values == -1).any():
            failures.append(f"{key} has missing/stale frames")
        if share < 0.5:
            failures.append(f"{key} running in only {share:.0%} of frames")
    head = df["observation.head.state"].map(list).tolist()
    tilt = [h[1] for h in head]
    print(f"  head tilt rad range {min(tilt):.3f}..{max(tilt):.3f}; hand right index cmd range "
          f"{min(a[2] for a in df['action.hand.right']):.3f}..{max(a[2] for a in df['action.hand.right']):.3f}")
    meta = (info.get("motion_acq") or info.get("handumi") or {})
    print(f"  frames {len(df)}; features {len(info.get('features', {}))}; sidecars meta {meta.get('sidecars')}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="arm4090")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--check-only", action="store_true", help="only re-check the last dataset")
    ap.add_argument("--arms-only", action="store_true", help="no head/hands (macq station --no-head --hands none)")
    args = ap.parse_args()
    out = LOGS / "dataset"
    if args.check_only:
        failures = check_dataset(out, args.seconds, 30, args.arms_only)
        print("FAIL: " + "; ".join(failures) if failures else "PASS")
        return 1 if failures else 0
    LOGS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    tcp = LOGS / "identity_tcp.yaml"
    tcp.write_text(IDENTITY_TCP)
    env = {**os.environ, "MACQ_STATION": args.station, "PYTHONUNBUFFERED": "1"}
    station_args = ["--no-head", "--hands", "none"] if args.arms_only else []
    master, slave = os.openpty()
    log_path = LOGS / "station.log"
    log = open(log_path, "w")
    # macq station --fake runs every producer and teleop-record in the foreground
    # on this pseudo-terminal, exactly as the assistant would use it.
    proc = subprocess.Popen([
        str(VENV / "macq"), "station", "--fake", *station_args, "--",
        "--controller-tcp-calibration", str(tcp), "--num-episodes", "1",
        "--episode-time-s", str(args.seconds), "--output-dir", str(out),
        "--no-preview", "--no-rerun", "--no-sounds", "--no-record-audio",
    ], stdin=slave, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
    os.close(slave)
    try:
        if not wait_for(log_path, "press Space to start episode", 180):
            print("recorder never became ready; see", log_path)
            return 1
        time.sleep(3.0)  # head anchors after 2 s of tracked HMD; fake hands enable on start
        os.write(master, b" ")
        proc.wait(timeout=args.seconds + 150)
    except subprocess.TimeoutExpired:
        print("station did not finish; see", log_path)
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        return 1
    print(f"macq station exit code {proc.returncode}")
    failures = check_dataset(out, args.seconds, 30, args.arms_only)
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
