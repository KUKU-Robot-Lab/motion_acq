"""Read-only station state for the console: Quest link, CAN, head port, RT, gloves, hand driver.

Nothing here writes to a device. Two background threads poll: a fast one
(adb, CAN, port holders, SenseCom, gloves) and a slow one for ROS 2 (topic list
and publisher counts on the station domain, where the s2r console runs the
RH56F1 EtherCAT driver). The head and hand processes also send their records
to the console's UDP ports (LatestUdp).
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

from motion_acq.console.units import FAKE_DOMAIN, SIDES, Station

FAST_PERIOD_S = 2.5
ROS_PERIOD_S = 6.0
QUEST_TCP = "tcp:65432"
VIEW_REVERSE = "tcp:8787"
GLOVE_DRIVER_PATTERN = "motion_acq_hand nova2.launch.py"


def run(cmd: list[str], timeout: float = 5.0, env: dict[str, str] | None = None) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=None if env is None else {**os.environ, **env}).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


# -- parsers (pure) ----------------------------------------------------------------

def parse_adb_devices(text: str) -> list[str]:
    """States of the listed devices ('device', 'unauthorized', 'no permissions ...')."""
    states = []
    for line in text.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) == 2:
            states.append(parts[1].strip())
    return states


def parse_ip_link(text: str) -> dict:
    up = "state UP" in text or ",UP," in text or ",UP>" in text
    return {"exists": bool(text.strip()), "up": up, "fd": "dbitrate 5000000" in text}


def parse_topic_info(text: str) -> tuple[int | None, int | None]:
    pubs = subs = None
    for line in text.splitlines():
        if line.startswith("Publisher count:"):
            pubs = int(line.split(":", 1)[1])
        elif line.startswith("Subscription count:"):
            subs = int(line.split(":", 1)[1])
    return pubs, subs


def parse_publisher_nodes(text: str) -> list[str]:
    """Node names of the publishers in `ros2 topic info -v` output."""
    nodes, name = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Node name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Endpoint type:") and name is not None:
            if line.split(":", 1)[1].strip() == "PUBLISHER":
                nodes.append(name)
            name = None
    return nodes


def parse_sections(text: str) -> dict[str, str]:
    """Split '@@name' delimited output of one bash call."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("@@"):
            current = line[2:].strip()
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def parse_ss_listener(text: str) -> dict | None:
    """Owner of a listening socket from `ss -ltnpH` output: {"name", "pid"} (None if nobody listens)."""
    for line in text.splitlines():
        if "users:((" not in line:
            continue
        users = line.split("users:((", 1)[1]
        name = users.split('"')[1] if users.count('"') >= 2 else "?"
        pid = users.split("pid=", 1)[1].split(",", 1)[0] if "pid=" in users else ""
        return {"name": name, "pid": int(pid) if pid.isdigit() else None}
    if text.strip():
        return {"name": "?", "pid": None}  # someone else's socket: ss shows no owner
    return None


def bt_connected(text: str) -> bool:
    return any(line.strip() == "Connected: yes" for line in text.splitlines())


# -- UDP records from head / hand processes ----------------------------------------------

class LatestUdp:
    """Keeps the newest JSON record sent to 127.0.0.1:port."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.latest: dict | None = None
        self.at = 0.0
        self.error = ""
        self._sock: socket.socket | None = None
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("127.0.0.1", port))
            self._sock.settimeout(0.5)
        except OSError as exc:
            self.error = f"UDP {port}: {exc}"
            self._sock = None
            return
        threading.Thread(target=self._run, daemon=True, name=f"console-udp-{port}").start()

    def _run(self) -> None:
        assert self._sock is not None
        while True:
            try:
                data = self._sock.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            try:
                record = json.loads(data)
            except ValueError:
                continue
            if isinstance(record, dict):
                self.latest, self.at = record, time.monotonic()

    def get(self, max_age_s: float = 2.0) -> dict | None:
        if self.latest is None or time.monotonic() - self.at > max_age_s:
            return None
        return {**self.latest, "age_s": round(time.monotonic() - self.at, 2)}


# -- probes ---------------------------------------------------------------------------------------

def adb_path() -> str:
    from motion_acq.quest_view.devtools import adb_path as _adb

    return _adb()


def quest_port_owner() -> dict | None:
    """Who listens on 127.0.0.1:65432: adb (the HandUMI forward), quest-view, the mock, or nobody."""
    return parse_ss_listener(run(["ss", "-ltnpH", "sport = :65432"], timeout=3))


def quest_state() -> dict:
    adb = adb_path()
    if not Path(adb).exists() and "/" in adb:
        return {"adb": False, "text": "adb 없음"}
    states = parse_adb_devices(run([adb, "devices"], timeout=4))
    out: dict = {"adb": True, "devices": states}
    if len(states) != 1 or states[0] != "device":
        out["text"] = ("Quest 가 USB 에 없다" if not states else
                       "헤드셋 안에서 USB 디버깅 허용 필요" if states[0] == "unauthorized" else
                       "adb 장치가 둘 이상" if len(states) > 1 else f"adb: {states[0]}")
        return out
    out["forward"] = QUEST_TCP in run([adb, "forward", "--list"], timeout=4)
    out["reverse"] = VIEW_REVERSE in run([adb, "reverse", "--list"], timeout=4)
    power = run([adb, "shell", "dumpsys power | grep -m1 mWakefulness="], timeout=4)
    out["awake"] = None if not power else "Awake" in power
    out["text"] = "연결됨"
    return out


def can_state(station: Station) -> dict:
    return {side: {"port": port, **parse_ip_link(run(["ip", "-details", "link", "show", port], timeout=3))}
            for side, port in station.can_ports.items()}


def can_holders() -> list[str]:
    from motion_acq.scripts.station import can_holders as _holders

    return _holders(run(["pgrep", "-af", "ros2_control_node|openarm.bimanual"], timeout=3))


def head_port(station: Station) -> dict:
    if not station.has_head:
        return {}
    from motion_acq.head.dynamixel import port_holders

    path = str(station.rig["head"].get("port", ""))
    if not path or not Path(path).exists():
        return {"path": path, "exists": False, "holders": []}
    return {"path": path, "exists": True, "holders": port_holders(path)}


def rt_state() -> dict:
    from motion_acq.cpu import ARM_STREAMER_FIFO, current_plan, format_cpulist, rt_limit

    plan = current_plan()
    limit = rt_limit()
    return {"rtprio": limit, "ok": limit >= ARM_STREAMER_FIFO,
            "plan": (f"일반 {format_cpulist(plan.general)}, EtherCAT {format_cpulist(plan.reserved)}"
                     if plan.rt else plan.note)}


def glove_state(station: Station) -> dict:
    if not station.hands:
        return {}
    from motion_acq.hand.nova2 import load_gloves
    from motion_acq.scripts.station import sensecom_started_at

    sensecom = sensecom_started_at()
    gloves = {}
    for side, glove in load_gloves().items():
        info = run(["bluetoothctl", "info", glove.mac], timeout=3) if sensecom else ""
        gloves[side] = {"serial": glove.serial, "connected": bt_connected(info), "topic": glove.topic}
    driver = [line for line in run(["pgrep", "-af", GLOVE_DRIVER_PATTERN], timeout=3).splitlines()
              if line.strip() and "pgrep" not in line]
    return {"sensecom_started_at": sensecom, "gloves": gloves,
            "driver_pids": [int(line.split()[0]) for line in driver]}


def ros_state(station: Station, mode: str) -> dict:
    """Hand driver topics and who commands them, on the domain the hand nodes use."""
    from motion_acq.console.units import UnitError, ros_argv

    if mode == "real":
        domain, env = station.ros_domain, {"ROS_DOMAIN_ID": station.ros_domain or "0"}
    else:
        domain, env = FAKE_DOMAIN, {"ROS_DOMAIN_ID": FAKE_DOMAIN, "ROS_LOCALHOST_ONLY": "1"}
    script = "echo @@list; ros2 topic list"
    for side in station.hands:
        for topic in ("angle_actual", "angle_set"):
            script += f"; echo @@{side}_{topic}; ros2 topic info -v /hand_{side}/{topic}"
    try:
        argv = list(ros_argv("bash -c " + shlex.quote(script)))
    except UnitError as exc:
        return {"domain": domain, "error": str(exc)}
    started = time.monotonic()
    sections = parse_sections(run(argv, timeout=25, env=env))
    if "list" not in sections:
        return {"domain": domain, "error": "ros2 응답 없음", "took_s": round(time.monotonic() - started, 1)}
    topics = set(sections["list"].split())
    hands = {}
    for side in station.hands:
        driver_pubs, _ = parse_topic_info(sections.get(f"{side}_angle_actual", ""))
        cmd_text = sections.get(f"{side}_angle_set", "")
        cmd_pubs, cmd_subs = parse_topic_info(cmd_text)
        hands[side] = {"driver": bool(driver_pubs), "angle_set_publishers": cmd_pubs or 0,
                       "angle_set_publisher_nodes": parse_publisher_nodes(cmd_text),
                       "angle_set_subscribers": cmd_subs or 0,
                       "ecat_status": f"/hand_{side}/ecat_status" in topics}
    gloves = {}
    if station.hands:
        from motion_acq.hand.nova2 import load_gloves

        gloves = {side: glove.topic in topics for side, glove in load_gloves().items() if side in SIDES}
    return {"domain": domain, "hands": hands, "glove_topics": gloves,
            "took_s": round(time.monotonic() - started, 1), "at": time.time()}


class Probes:
    """Background pollers; snapshot() returns the latest of everything."""

    def __init__(self, station: Station, mode_getter) -> None:
        self.station = station
        self.mode_getter = mode_getter
        self.data: dict = {"rt": rt_state()}
        self._stop = threading.Event()
        self._ros_kick = threading.Event()
        for name, target in (("fast", self._fast_loop), ("ros", self._ros_loop)):
            threading.Thread(target=target, daemon=True, name=f"console-probe-{name}").start()

    def _guard(self, key: str, fn) -> None:
        try:
            self.data[key] = fn()
        except Exception as exc:  # noqa: BLE001 - a probe must never stop the console
            self.data[key] = {"error": f"{type(exc).__name__}: {exc}"}

    def _fast_loop(self) -> None:
        while not self._stop.is_set():
            self._guard("quest", quest_state)
            self._guard("quest_port", quest_port_owner)
            self._guard("can", lambda: can_state(self.station))
            self._guard("can_holders", can_holders)
            self._guard("head_port", lambda: head_port(self.station))
            self._guard("gloves", lambda: glove_state(self.station))
            self.data["fast_at"] = time.time()
            self._stop.wait(FAST_PERIOD_S)

    def _ros_loop(self) -> None:
        if not self.station.hands:
            return
        while not self._stop.is_set():
            self._guard("ros", lambda: ros_state(self.station, self.mode_getter()))
            self._ros_kick.wait(ROS_PERIOD_S)
            self._ros_kick.clear()

    def refresh_ros(self) -> None:
        self._ros_kick.set()

    def close(self) -> None:
        self._stop.set()
        self._ros_kick.set()

    def snapshot(self) -> dict:
        return dict(self.data)
