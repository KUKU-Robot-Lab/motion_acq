#!/usr/bin/env python3
"""Bring up one station: head + hands as their own processes, the recorder in front.

    MACQ_STATION=arm4090 macq station --fake -- --num-episodes 3
    MACQ_STATION=arm4090 macq station --real --user op1 -- --num-episodes 10
    MACQ_STATION=arm5080 macq station --real -- --num-episodes 10     # arms only

The station rig decides which streams exist (recording.sidecars). Each
subsystem runs in its own process group: a head or hand failure never stops
the arms, the recorder just rejects the episode (stale stream). teleop-record
runs in the foreground so the assistant's keyboard (Space / R / Q / Esc)
reaches it; everything after ``--`` is passed to it.

--fake : mock Quest, fake head bus, fake hands (ROS domain 177, localhost),
         simulated OpenArm SDK. Nothing touches hardware.
--real : read-only preflight first (Quest link, CAN up and not held by s2r,
         head port free, RH56F1 driver and glove topics, calibration files);
         any failure stops before a process starts. Hands start disabled and
         are enabled only after the operator presses Enter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from motion_acq.config import STATION_ENV, load_rig_config, station_rig_config

ROOT = Path(__file__).resolve().parents[3]
VENV_BIN = Path(sys.executable).parent
FAKE_DOMAIN = "177"


@dataclass
class Plan:
    station: str
    rig: dict
    mode: str
    streams: dict[str, int]
    user: str = ""
    log_dir: Path = field(default_factory=Path)

    @property
    def hands(self) -> list[str]:
        return [name.split("_", 1)[1] for name in self.streams if name.startswith("hand_")]


def _ros_command(command: str, *, fake: bool) -> list[str]:
    distro = os.environ.get("ROS_DISTRO") or next(
        (d for d in ("humble", "jazzy") if Path(f"/opt/ros/{d}/setup.bash").exists()), ""
    )
    if not distro:
        raise SystemExit("no ROS 2 under /opt/ros (hands need it)")
    rc_ws = os.environ.get("ROBOT_CONTROL_WS", str(Path.home() / "rl_ws/robot_control/ros_ws/install"))
    env = f"export ROS_DOMAIN_ID={FAKE_DOMAIN} ROS_LOCALHOST_ONLY=1; " if fake else ""
    script = (
        f"set +u; source /opt/ros/{distro}/setup.bash; source {rc_ws}/setup.bash; "
        f"source {ROOT}/ros_ws/install/setup.bash; {env}exec {command}"
    )
    return ["bash", "-c", script]


# -- preflight (read-only) ------------------------------------------------------


def _check(ok: bool, label: str, detail: str, problems: list[str]) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        problems.append(label)


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def preflight(plan: Plan) -> list[str]:
    from motion_acq.head.config import load_head_config
    from motion_acq.head.dynamixel import port_holders
    from motion_acq.real.openarm.driver import load_openarm_settings
    from motion_acq.robots.registry import load_embodiment

    problems: list[str] = []
    domain = os.environ.get("ROS_DOMAIN_ID") or "0 (unset)"
    print(f"Preflight ({plan.station}, read-only; ROS_DOMAIN_ID={domain}, must match the RH56F1/glove nodes):")
    quest = (plan.rig.get("meta_quest") or {}).get("connection") or {}
    if quest.get("quest_ip") == "127.0.0.1":
        adb = shutil.which("adb") or str(Path.home() / "opt/platform-tools/adb")
        forwards = _run([adb, "forward", "--list"])
        linked = "tcp:65432" in forwards
        _check(linked, "Quest USB link", "adb forward tcp:65432" if linked else
               "no tcp:65432 forward (scripts/quest_usb.sh status / forward)", problems)
    robot = str((plan.rig.get("recording") or {}).get("robot") or "openarmv1")
    runtime = load_embodiment(robot)
    settings = load_openarm_settings(
        station_rig_config(plan.station), runtime.config.real_options, None, robot_name=robot
    )
    for port in (settings.left_port, settings.right_port):
        details = _run(["ip", "-details", "link", "show", port])
        up = "state UP" in details or ",UP," in details
        fd = "dbitrate 5000000" in details
        _check(up and fd, f"CAN {port}", "UP, FD 1M/5M" if up and fd else "down or not FD", problems)
    holders = _run(["pgrep", "-af", "ros2_control_node|openarm.bimanual"]).strip()
    _check(not holders, "CAN not held by s2r", "free" if not holders else holders.splitlines()[0], problems)
    if "head" in plan.streams:
        head = load_head_config(station_rig_config(plan.station), allow_fake_default=False)
        held = port_holders(head.port) if Path(head.port).exists() else None
        _check(held == [], "head serial port", head.port if held == [] else
               ("missing" if held is None else f"held by PID {held}"), problems)
    if plan.hands:
        topics = _run(_ros_command("ros2 topic list", fake=False), timeout=20).split()
        for side in plan.hands:
            _check(f"/hand_{side}/angle_actual" in topics, f"RH56F1 {side} driver",
                   f"/hand_{side}/angle_actual", problems)
            glove = _glove(plan, side)
            serial = str(glove.get("glove_serial") or "")
            topic = f"/senseglove/glove{serial}/{'rh' if side == 'right' else 'lh'}/senseglove_states"
            _check(bool(serial) and topic in topics, f"Nova 2 {side} glove",
                   topic if serial else "hands.<side>.glove_serial not set in the rig", problems)
            cal = _calibration(plan, side)
            _check(cal.exists(), f"{side} hand calibration", str(cal), problems)
    return problems


def _glove(plan: Plan, side: str) -> dict:
    return ((plan.rig.get("hands") or {}).get(side)) or {}


def _calibration(plan: Plan, side: str) -> Path:
    if plan.mode == "fake":
        return ROOT / "logs" / "hand" / f"fake_calibration_{side}.yaml"
    return ROOT / "configs" / "hands" / "calibration" / f"{plan.user}_{side}.yaml"


# -- processes ------------------------------------------------------------------


class Group:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.procs: list[tuple[str, subprocess.Popen]] = []
        self._stop = threading.Event()

    def start(self, name: str, cmd: list[str], env: dict | None = None) -> None:
        log = open(self.log_dir / f"{name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=ROOT, env=env)
        self.procs.append((name, proc))
        print(f"  started {name} (PID {proc.pid}), log {self.log_dir / (name + '.log')}")

    def watch(self) -> None:
        def run() -> None:
            reported: set[str] = set()
            while not self._stop.wait(1.0):
                for name, proc in self.procs:
                    if proc.poll() is not None and name not in reported:
                        reported.add(name)
                        sys.stderr.write(
                            f"\r\n[station] {name} exited ({proc.returncode}); its stream is now stale. "
                            f"See {self.log_dir / (name + '.log')}\r\n"
                        )

        threading.Thread(target=run, name="station-watch", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        for _name, proc in reversed(self.procs):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        deadline = time.monotonic() + 15
        for _name, proc in reversed(self.procs):
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)


def start_producers(plan: Plan, group: Group, env: dict) -> None:
    fake = plan.mode == "fake"
    if fake:
        group.start("mock_quest", [str(VENV_BIN / "python"), "-m", "motion_acq.tracking.mock_quest_sender",
                                   "--hmd-yaw-amp-deg", "20", "--hmd-pitch-amp-deg", "8"], env)
        time.sleep(1.0)
    if "head" in plan.streams:
        group.start("head", [str(VENV_BIN / "macq"), "head", "--backend", "fake" if fake else "real",
                             "--udp-target", f"127.0.0.1:{plan.streams['head']}",
                             "--log-dir", str(plan.log_dir)], env)
    for side in plan.hands:
        udp = f"127.0.0.1:{plan.streams['hand_' + side]}"
        cal = _calibration(plan, side)
        if fake:
            cmd = (f"ros2 launch motion_acq_hand fake_hand.launch.py side:={side} calibration:={cal} "
                   f"udp_target:={udp}")
        else:
            serial = _glove(plan, side).get("glove_serial")
            cmd = (f"ros2 run motion_acq_hand hand_node --ros-args -r __node:=motion_acq_hand_{side} "
                   f"-p side:={side} -p glove_serial:={serial} -p calibration:={cal} -p udp_target:={udp} "
                   f"-p log_dir:={plan.log_dir}")
        group.start(f"hand_{side}", _ros_command(cmd, fake=fake), env)


def enable_hands(plan: Plan) -> None:
    for side in plan.hands:
        subprocess.run(_ros_command(
            f"ros2 topic pub --once /motion_acq/hand_{side}/enable std_msgs/msg/Bool '{{data: true}}'",
            fake=False), capture_output=True, timeout=30)
        print(f"  hand {side}: enable sent (check its log for 'enabled' or a refusal)")


def disable_hands(plan: Plan) -> None:
    for side in plan.hands:
        subprocess.run(_ros_command(
            f"ros2 topic pub --once /motion_acq/hand_{side}/enable std_msgs/msg/Bool '{{data: false}}'",
            fake=False), capture_output=True, timeout=30)


def recorder_command(plan: Plan, extra: list[str]) -> list[str]:
    cmd = [str(VENV_BIN / "macq"), "teleop-record", "--device", "meta", "--space-start", "--skip-feetech"]
    if not (plan.rig.get("cameras") or {}):
        cmd.append("--skip-cameras")
    if plan.streams:
        for name, port in plan.streams.items():
            cmd += ["--sidecar", f"{name}={port}"]
    else:
        cmd.append("--no-sidecars")
    if plan.mode == "fake":
        cmd += ["--fake-robot", "--quest-ip", "127.0.0.1"]
    if "--output-dir" not in extra:
        out = ROOT / "outputs" / f"{plan.station}_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd += ["--output-dir", str(out)]
    return cmd + extra


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--user", default="", help="hand calibration owner (configs/hands/calibration/<user>_<side>.yaml)")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--hands", default=None, help="right,left | right | left | none (default: the rig streams)")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)

    station = os.environ.get(STATION_ENV, "")
    if not station:
        raise SystemExit(f"set {STATION_ENV} (arm4090 | arm5080)")
    rig = load_rig_config(station_rig_config(station))
    streams = {k: int(v) for k, v in ((rig.get("recording") or {}).get("sidecars") or {}).items()}
    if args.no_head:
        streams.pop("head", None)
    if args.hands is not None:
        wanted = set() if args.hands == "none" else set(args.hands.split(","))
        streams = {k: v for k, v in streams.items() if not k.startswith("hand_") or k[5:] in wanted}
    plan = Plan(station, rig, "fake" if args.fake else "real", streams, args.user)
    if plan.mode == "real" and plan.hands and not plan.user:
        raise SystemExit("--user is required with hands (calibration file owner)")
    plan.log_dir = ROOT / "logs" / "station" / time.strftime("%Y%m%d_%H%M%S")
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Station {station} ({plan.mode}): streams {streams or 'none (arms only)'}")

    if plan.mode == "real":
        problems = preflight(plan)
        if problems:
            raise SystemExit(f"Preflight failed: {', '.join(problems)}. Nothing was started.")
        if args.preflight_only:
            print("Preflight passed.")
            return
    else:
        missing = [s for s in plan.hands if not _calibration(plan, s).exists()]
        if missing:
            raise SystemExit(f"run scripts/fake_hand_check.sh {missing[0]} first (fake calibration)")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", STATION_ENV: station}
    if plan.mode == "fake":
        env["MACQ_FAKE_ROBOT_START"] = "home"
    group = Group(plan.log_dir)
    try:
        start_producers(plan, group, env)
        group.watch()
        if plan.mode == "real" and plan.hands:
            input("Hands are disabled. Press Enter to enable them (they start from their measured pose) ")
            enable_hands(plan)
        print("Recorder (Space start/save, R reset, Q finish, Esc stop):")
        result = subprocess.run(recorder_command(plan, extra), cwd=ROOT, env=env)
        print(f"teleop-record exited ({result.returncode}).")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if plan.mode == "real" and plan.hands:
            disable_hands(plan)
        group.stop()
        print(f"Station stopped. Logs: {plan.log_dir}")


if __name__ == "__main__":
    main()
