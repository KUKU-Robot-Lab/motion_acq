#!/usr/bin/env python3
"""Fake end-to-end recording (no hardware): arms + head + both hands -> one LeRobot dataset.

    .venv/bin/python scripts/fake_record_check.py [--station arm4090] [--seconds 8]

Starts the mock Quest (moving HMD), `macq head` on its fake bus, the fake hand
chains for both sides (ROS_DOMAIN_ID=177, localhost only), then
`macq teleop-record --fake-robot` in a pseudo-terminal, presses Space once to
start, lets --episode-time-s save the episode, and checks the dataset. Every
process it starts is its own child and is stopped by PID at the end.
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
PORTS = {"head": 47101, "hand_right": 47111, "hand_left": 47112}
IDENTITY_TCP = """calibration:
  frame_convention: pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion
  controller_to_gripper_tcp:
    left: {position: [0.0, 0.0, 0.0], quaternion: [0.0, 0.0, 0.0, 1.0]}
    right: {position: [0.0, 0.0, 0.0], quaternion: [0.0, 0.0, 0.0, 1.0]}
"""


def ros_env_cmd(command: str) -> list[str]:
    distro = os.environ.get("ROS_DISTRO") or next(
        d for d in ("humble", "jazzy") if Path(f"/opt/ros/{d}/setup.bash").exists()
    )
    rc_ws = os.environ.get("ROBOT_CONTROL_WS", str(Path.home() / "rl_ws/robot_control/ros_ws/install"))
    script = (
        f"set +u; source /opt/ros/{distro}/setup.bash; source {rc_ws}/setup.bash; "
        f"source {ROOT}/ros_ws/install/setup.bash; export ROS_DOMAIN_ID=177 ROS_LOCALHOST_ONLY=1; "
        f"exec {command}"
    )
    return ["bash", "-c", script]


class Procs:
    def __init__(self) -> None:
        self.items: list[tuple[str, subprocess.Popen]] = []

    def start(self, name: str, cmd: list[str], **kw) -> subprocess.Popen:
        log = open(LOGS / f"{name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                cwd=ROOT, **kw)
        self.items.append((name, proc))
        return proc

    def stop_all(self) -> None:
        for _name, proc in reversed(self.items):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)  # own session: launch + its nodes
        deadline = time.monotonic() + 15
        for _name, proc in reversed(self.items):
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)


def wait_for(path: Path, text: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text(errors="replace"):
            return True
        time.sleep(0.2)
    return False


def check_dataset(out: Path, seconds: float, fps: int) -> list[str]:
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
    needed = ["observation.state", "action", "observation.head.state", "action.head",
              "observation.hand.right.state", "action.hand.right", "observation.glove.right.angles",
              "observation.hand.left.state", "action.hand.left", "observation.glove.left.angles"]
    missing = [k for k in needed if k not in df.columns]
    if missing:
        failures.append(f"missing columns {missing}")
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
    args = ap.parse_args()
    if args.check_only:
        failures = check_dataset(LOGS / "dataset", args.seconds, 30)
        print("FAIL: " + "; ".join(failures) if failures else "PASS")
        return 1 if failures else 0
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "dataset"
    shutil.rmtree(out, ignore_errors=True)
    tcp = LOGS / "identity_tcp.yaml"
    tcp.write_text(IDENTITY_TCP)
    cal = {side: ROOT / "logs" / "hand" / f"fake_calibration_{side}.yaml" for side in ("right", "left")}
    env = {**os.environ, "MACQ_STATION": args.station, "MACQ_FAKE_ROBOT_START": "home",
           "PYTHONUNBUFFERED": "1"}
    procs = Procs()
    try:
        for side in ("right", "left"):
            if not cal[side].exists():
                print(f"run scripts/fake_hand_check.sh {side} first (needs {cal[side]})")
                return 2
        procs.start("mock_quest", [str(VENV / "python"), "-m", "motion_acq.tracking.mock_quest_sender",
                                   "--hmd-yaw-amp-deg", "20", "--hmd-pitch-amp-deg", "8"], env=env)
        time.sleep(1.5)
        procs.start("head", [str(VENV / "macq"), "head", "--udp-target", f"127.0.0.1:{PORTS['head']}",
                             "--quest-ip", "127.0.0.1", "--log-dir", str(LOGS)], env=env)
        for side in ("right", "left"):
            procs.start(f"hand_{side}", ros_env_cmd(
                f"ros2 launch motion_acq_hand fake_hand.launch.py side:={side} calibration:={cal[side]} "
                f"udp_target:=127.0.0.1:{PORTS['hand_' + side]}"))
        master, slave = os.openpty()
        rec_log = LOGS / "teleop_record.log"
        procs.start("teleop_record", [
            str(VENV / "macq"), "teleop-record", "--device", "meta", "--fake-robot", "--skip-feetech",
            "--space-start", "--skip-cameras", "--quest-ip", "127.0.0.1",
            "--controller-tcp-calibration", str(tcp), "--num-episodes", "1",
            "--episode-time-s", str(args.seconds), "--output-dir", str(out),
            "--no-preview", "--no-rerun", "--no-sounds", "--no-record-audio",
        ], env=env, stdin=slave)
        os.close(slave)
        if not wait_for(rec_log, "press Space to start episode", 180):
            print("recorder never became ready; see", rec_log)
            return 1
        time.sleep(3.0)  # head anchors after 2 s of tracked HMD; hands enable on start
        os.write(master, b" ")
        recorder = procs.items[-1][1]
        try:
            recorder.wait(timeout=args.seconds + 120)
        except subprocess.TimeoutExpired:
            print("recorder did not finish; see", rec_log)
            return 1
        print(f"teleop-record exit code {recorder.returncode}")
    finally:
        procs.stop_all()
    failures = check_dataset(out, args.seconds, 30)
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
