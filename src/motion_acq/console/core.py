"""The console's model: which units run, what the operator confirmed, what the window shows.

Every operator action goes through here and is appended to
logs/console/intents.jsonl. A unit that moves the robot (real mode) starts
only from a confirmation token: the first request returns need_confirm with
the exact command and a token, the second (confirm + token) starts exactly
that stored command after the blockers are checked again. A changed or
expired command is asked again. What blocks a start lives in gates.py.
"""

from __future__ import annotations

import collections
import json
import os
import secrets
import signal
import string
import threading
import time
from pathlib import Path

from motion_acq.console import gates, phases, units
from motion_acq.console.probes import LatestUdp, Probes, adb_path, ros_state, run
from motion_acq.console.supervisor import Supervisor, _same_process, left_behind, process_start_epoch, \
    stop_left_behind
from motion_acq.console.units import ROOT, Launch, Station, UnitError

SPACE_UNITS = ("record", "arm", "head")
ALLOWED_KEYS = {" ", "r", "q", "\x1b", "\n", "y\n", "n\n"}
CONFIRM_TTL_S = 120.0
DEFAULT_SETTINGS = {"user": "", "arm_side": "right", "scale": units.DEFAULT_SCALE,
                    "task": "teleop demonstration", "episodes": 10}


def same_command(a: Launch, b: Launch) -> bool:
    """Same program, options and environment; a record --output-dir (a timestamp) may differ."""
    def masked(launch: Launch) -> list[str]:
        argv = list(launch.argv)
        return [("<out>" if i and argv[i - 1] == "--output-dir" else arg) for i, arg in enumerate(argv)]
    return a.key == b.key and masked(a) == masked(b) and a.env == b.env


class Console:
    def __init__(self, station: Station, mode: str, *, log_root: Path | None = None,
                 probes: bool = True, listen: bool = True) -> None:
        if mode not in ("real", "fake"):
            raise UnitError(f"mode must be real or fake, not {mode!r}")
        self.station = station
        self.mode = mode
        self.closing = False
        self.log_root = log_root or ROOT / "logs" / "console"
        self.session_dir = self.log_root / time.strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        registry = self.log_root / "running.json"
        self.left_behind = left_behind(registry)
        self.sup = Supervisor(self.session_dir, registry)
        self.sup.foreign = self.left_behind  # kept in the registry while alive
        self.settings = self._load_settings()
        self.space_target: str | None = None
        self.events: collections.deque = collections.deque(maxlen=40)
        self.jobs: dict[str, dict] = {}
        self.pending: dict[str, tuple[str, Launch, float]] = {}  # key -> (token, launch, time)
        self.lock = threading.RLock()
        self.probes = Probes(station, lambda: self.mode) if probes else None
        self.head_udp = LatestUdp(units.CONSOLE_HEAD_PORT) if listen and station.has_head else None
        self.hand_udp = ({side: LatestUdp(units.CONSOLE_HAND_PORTS[side]) for side in station.hands}
                         if listen else {})
        if self.left_behind:
            self.event("bad", "이전 콘솔이 남긴 프로세스: " + ", ".join(
                f"{e['key']} (PID {e['pid']})" for e in self.left_behind))

    # -- bookkeeping -------------------------------------------------------------------
    def event(self, level: str, text: str) -> None:
        self.events.append({"t": time.time(), "level": level, "text": text})

    def intent(self, action: str, **fields) -> None:
        record = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "station": self.station.name, "mode": self.mode,
                  "action": action, **fields}
        try:
            with (self.log_root / "intents.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # never fail an action (a unit may already run) on the log
            self.event("warn", f"intents.jsonl 쓰기 실패: {exc}")

    def probe(self, name: str) -> dict:
        value = self.probe_raw(name)
        return value if isinstance(value, dict) else {}

    def probe_raw(self, name: str):
        return self.probes.snapshot().get(name) if self.probes else None

    def left_keys(self) -> set[str]:
        self.left_behind[:] = [e for e in self.left_behind if _same_process(e)]
        return {e["key"] for e in self.left_behind}

    def _settings_path(self) -> Path:
        return self.log_root / "settings.json"

    def _load_settings(self) -> dict:
        try:
            saved = json.loads(self._settings_path().read_text())
        except (OSError, ValueError):
            saved = {}
        return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}

    def update_settings(self, values: dict) -> dict:
        with self.lock:
            clean = dict(self.settings)
            if "user" in values:
                user = str(values["user"]).strip()
                if user:
                    units.calibration_file(user, "right")  # validates
                clean["user"] = user
            if "arm_side" in values:
                clean["arm_side"] = units.check_side(str(values["arm_side"]), units.ARM_SIDES)
            if "scale" in values:
                clean["scale"] = units.check_scale(float(values["scale"]))
            if "task" in values:
                clean["task"] = str(values["task"])[:200]
            if "episodes" in values:
                clean["episodes"] = max(1, min(500, int(float(values["episodes"]))))
            self.settings = clean
            self._settings_path().write_text(json.dumps(clean, ensure_ascii=False, indent=1))
            return clean

    def set_mode(self, mode: str) -> None:
        if mode not in ("real", "fake"):
            raise UnitError("mode must be real or fake")
        with self.lock:
            busy = self.sup.running()
            if busy and mode != self.mode:
                raise UnitError("실행 중인 것을 모두 정지한 뒤 바꿀 수 있다: " + ", ".join(busy))
            self.mode = mode
            self.pending.clear()  # a command confirmed for the other mode must not start
        self.intent("mode", mode=mode)
        if self.probes:
            self.probes.refresh_ros()

    # -- building -----------------------------------------------------------------------
    def build(self, key: str) -> Launch:
        st, mode, s = self.station, self.mode, self.settings
        if key == "quest_view":
            return units.quest_view(st)
        if key == "mock_quest":
            if mode == "real":
                raise UnitError("가짜 Quest 는 fake 모드에서만")
            return units.mock_quest(st)
        if key == "head":
            return units.head(st, mode)
        if key == "arm":
            return units.arm(st, mode, str(s["arm_side"]), float(s["scale"]))
        if key == "record":
            streams = units.record_streams(st, set(self.sup.running()))
            out = ROOT / "outputs" / f"{st.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            return units.record(st, mode, str(s["arm_side"]), float(s["scale"]), streams=streams,
                                task=str(s["task"]), episodes=int(s["episodes"]), output_dir=out)
        if key.startswith("hand_"):
            return units.hand(st, mode, key[5:], str(s["user"]), self.session_dir / "hand")
        if key.startswith("calib_"):
            return units.calibrate(st, mode, key[6:], str(s["user"]))
        raise UnitError(f"unknown unit {key!r}")

    def blockers(self, key: str) -> list[str]:
        return gates.blockers(self, key)

    # -- starting -------------------------------------------------------------------------
    def start(self, key: str, opts: dict | None = None, *, confirm: bool = False, token: str = "") -> dict:
        with self.lock:
            if opts:
                self.update_settings(opts)  # the blockers read the same side / scale
            problems = self.blockers(key)
            if problems:
                return {"ok": False, "error": " / ".join(problems)}
            return self._gate(self.build(key), confirm=confirm, token=token)

    def _gate(self, launch: Launch, *, confirm: bool, token: str) -> dict:
        """Start launch, or ask for (and later honour) the operator's confirmation of its exact command."""
        if launch.moves_robot:
            pending = self.pending.get(launch.key)
            approved = (pending is not None and confirm and bool(token)
                        and time.time() - pending[2] < CONFIRM_TTL_S and same_command(pending[1], launch)
                        and secrets.compare_digest(pending[0], token))
            if pending is None or not approved:
                new_token = secrets.token_hex(8)
                self.pending[launch.key] = (new_token, launch, time.time())
                reply = {"ok": False, "need_confirm": True, "token": new_token, "title": launch.title,
                         "summary": launch.summary, "command": launch.display()}
                if confirm:
                    reply["summary"] = "확인한 뒤 명령이 바뀌었거나 시간이 지났다. 다시 확인할 것. " + launch.summary
                return reply
            launch = pending[1]  # exactly what the operator saw
            del self.pending[launch.key]
        return self._spawn(launch, confirmed=launch.moves_robot)

    def _spawn(self, launch: Launch, *, confirmed: bool = False) -> dict:
        with self.lock:
            if self.closing:
                return {"ok": False, "error": "콘솔이 종료 중이다"}
            try:
                child = self.sup.start(launch.key, list(launch.argv), cwd=ROOT, env={**os.environ, **launch.env},
                                       pty=launch.pty)
            except (RuntimeError, OSError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}
            if launch.key in SPACE_UNITS:
                self.space_target = launch.key
        self.intent("start", key=launch.key, confirmed=confirmed, command=launch.display(), pid=child.pid)
        self.event("info", f"{launch.title} 시작 (PID {child.pid})")
        return {"ok": True, "pid": child.pid}

    # -- stopping and keys ------------------------------------------------------------------
    def stop(self, key: str) -> dict:
        if not self.sup.is_running(key):
            return {"ok": False, "error": f"{key} 는 실행 중이 아니다"}
        if not self.sup.stop(key):
            return {"ok": False, "error": "이미 정지 중: 안전 자세로 가는 동안 기다릴 것"}
        self.intent("stop", key=key)
        self.event("info", f"{key} 정지 요청(SIGINT, 안전 자세 복귀)")
        return {"ok": True}

    def force_stop(self, key: str, *, confirm: bool) -> dict:
        if not confirm:
            return {"ok": False, "error": "강제 종료는 확인이 필요하다"}
        if not self.sup.is_running(key):
            return {"ok": False, "error": f"{key} 는 실행 중이 아니다"}
        self.intent("force_stop", key=key)
        self.event("bad", f"{key} 강제 종료(SIGTERM → SIGKILL): 안전 자세 복귀 없음")
        threading.Thread(target=self.sup.force_stop, args=(key,), daemon=True).start()
        return {"ok": True}

    def send_key(self, key: str, text: str) -> dict:
        line = text.endswith("\n") and len(text) <= 81 and all(c in string.printable for c in text)
        if text not in ALLOWED_KEYS and not line:
            return {"ok": False, "error": "보낼 수 없는 키"}
        try:
            self.sup.send_key(key, text)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        self.intent("key", key=key, text=repr(text))
        return {"ok": True}

    def stop_all(self) -> dict:
        stopped = [key for key in self.sup.running() if self.sup.stop(key)]
        self.intent("stop_all", keys=stopped)
        self.event("warn", "모두 정지 요청: " + (", ".join(stopped) or "없음"))
        return {"ok": True, "stopped": stopped}

    def set_space_target(self, key: str | None) -> dict:
        if key is not None and key not in SPACE_UNITS:
            return {"ok": False, "error": "Space 대상이 될 수 없다"}
        self.space_target = key
        return {"ok": True}

    # -- tasks and sequences --------------------------------------------------------------------
    def action(self, name: str, *, confirm: bool = False, token: str = "") -> dict:
        kind, _, arg = name.partition(":")
        if kind == "quest" and arg in ("app", "view"):
            return self._job(f"quest_{arg}", self._quest_app if arg == "app" else self._quest_view)
        if kind == "left_behind_stop":
            return self._stop_left_behind(int(arg))
        if kind == "glove_driver_stop":
            return self._stop_glove_driver()
        with self.lock:
            if kind == "quest":
                return self._gate(units.quest_task(self.station, arg), confirm=confirm, token=token)
            if kind == "glove":
                return self._gate(units.glove_task(self.station, self.mode, arg), confirm=confirm, token=token)
            if kind == "check":
                return self._spawn(units.station_check(self.station, str(self.settings["user"])))
        if kind in ("hand_on", "hand_off"):
            return self._hand_switch(units.check_side(arg), kind == "hand_on", confirm=confirm, token=token)
        return {"ok": False, "error": f"unknown action {name!r}"}

    def _hand_switch(self, side: str, on: bool, *, confirm: bool, token: str) -> dict:
        if not self.sup.is_running(f"hand_{side}"):
            return {"ok": False, "error": "손 노드를 먼저 시작할 것"}
        if on:
            ros = ros_state(self.station, self.mode)  # read now, not the last poll
            ros["at"] = time.time()
            problems = gates.hand_blockers(self, side, ros, starting=False)
            if problems:
                return {"ok": False, "error": " / ".join(problems)}
        with self.lock:
            return self._gate(units.hand_enable(self.station, self.mode, side, on), confirm=confirm, token=token)

    def _job(self, name: str, fn) -> dict:
        with self.lock:
            job = self.jobs.get(name)
            if job and job.get("state") == "running":
                return {"ok": False, "error": "이미 진행 중"}
            if self.closing:
                return {"ok": False, "error": "콘솔이 종료 중이다"}
            self.jobs[name] = {"state": "running", "text": "", "at": time.time()}

        def body():
            try:
                text = fn()
                self.jobs[name] = {"state": "ok", "text": text, "at": time.time()}
            except Exception as exc:  # noqa: BLE001 - shown to the operator
                self.jobs[name] = {"state": "failed", "text": str(exc), "at": time.time()}
                self.event("bad", f"{name}: {exc}")

        self.intent("job", name=name)
        threading.Thread(target=body, daemon=True, name=f"console-job-{name}").start()
        return {"ok": True}

    def _wait_exit(self, key: str, timeout_s: float) -> int:
        deadline = time.time() + timeout_s
        while self.sup.is_running(key) and time.time() < deadline:
            time.sleep(0.2)
        if self.sup.is_running(key):
            raise RuntimeError(f"{key} 가 {timeout_s:.0f} s 안에 끝나지 않았다")
        child = self.sup.get(key)
        return -1 if child is None else int(child.proc.returncode or 0)

    def _quest_users_check(self) -> None:
        if any(self.sup.is_running(k) for k in gates.QUEST_PRODUCERS):
            raise RuntimeError("머리/팔/녹화가 Quest 를 쓰는 중: 먼저 정지")

    def _quest_app(self) -> str:
        """HandUMI app: quest-view (or the mock) must let go of local port 65432 first."""
        self._quest_users_check()
        if self.sup.is_running("mock_quest"):
            raise RuntimeError("가짜 Quest 가 65432 를 쓰는 중: 먼저 정지")
        if self.sup.is_running("quest_view"):
            self.sup.stop("quest_view")
            self._wait_exit("quest_view", 10)
        if gates.quest_source(self) == "external":
            raise RuntimeError(gates.external_quest(self) + ": 그 프로그램을 먼저 끌 것")
        result = self._spawn(units.quest_task(self.station, "app"))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "quest_usb.sh app"))
        rc = self._wait_exit("task_quest_app", 40)
        if rc != 0:
            raise RuntimeError(f"quest_usb.sh app 실패 (rc {rc}): 로그 확인")
        return "HandUMI 앱 연결됨"

    def _quest_view(self) -> str:
        """Head camera page: drop the app forward, start quest-view, open the page and enter VR."""
        self._quest_users_check()
        with self.lock:
            if not self.sup.is_running("quest_view"):
                problems = [p for p in self.blockers("quest_view")]
                if gates.quest_source(self) == "external":
                    problems.append(gates.external_quest(self))
                if problems:
                    raise RuntimeError(" / ".join(problems))
                run([adb_path(), "forward", "--remove", "tcp:65432"], timeout=5)
                result = self._spawn(units.quest_view(self.station))
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "quest-view"))
        deadline = time.time() + 15
        while not any("page http://localhost" in line for line in self.sup.log_tail("quest_view", 50)):
            if not self.sup.is_running("quest_view") or time.time() > deadline:
                raise RuntimeError("quest-view 가 뜨지 않았다: 로그 확인")
            time.sleep(0.3)
        result = self._spawn(units.quest_task(self.station, "view"))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "quest_usb.sh view"))
        if self._wait_exit("task_quest_view", 60) != 0:
            raise RuntimeError("페이지는 열렸지만 VR 이 PC 에서 시작되지 않았다: 헤드셋을 쓰고 [VR 시작]")
        return "헤드셋 영상 + VR 시작됨"

    def _stop_glove_driver(self) -> dict:
        from motion_acq.console.probes import GLOVE_DRIVER_PATTERN

        stopped = []
        for pid in (self.probe("gloves").get("driver_pids") or []):
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if GLOVE_DRIVER_PATTERN.split()[0] not in cmdline or process_start_epoch(pid) is None:
                continue
            try:
                os.killpg(pid, signal.SIGINT)
                stopped.append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        self.intent("glove_driver_stop", pids=stopped)
        return {"ok": bool(stopped), "error": "" if stopped else "장갑 드라이버가 안 보인다"}

    def _stop_left_behind(self, pid: int) -> dict:
        entry = next((e for e in self.left_behind if e["pid"] == pid), None)
        if entry is None:
            return {"ok": False, "error": "목록에 없는 PID(이미 끝났을 수 있다)"}
        if entry.get("stopping"):
            return {"ok": False, "error": "이미 정지 중: 안전 자세로 가는 동안 기다릴 것"}
        stop_left_behind(entry)
        entry["stopping"] = True  # stays listed (and blocking) until it exits
        self.intent("left_behind_stop", pid=pid, key=entry["key"])
        return {"ok": True}

    # -- state for the window ---------------------------------------------------------------------
    def unit_keys(self) -> list[str]:
        keys = ["quest_view"] + (["mock_quest"] if self.mode == "fake" else [])
        keys += (["head"] if self.station.has_head else []) + ["arm", "record"]
        for side in self.station.hands:
            keys += [f"hand_{side}", f"calib_{side}"]
        return keys

    def _unit(self, key: str) -> dict:
        snap = self.sup.snapshot(key)
        tail = self.sup.log_tail(key, 120)
        snap["phase"] = phases.phase(key, tail) if snap["running"] or snap["rc"] is not None else None
        snap["prompt"] = phases.prompt(tail, snap.get("partial", "")) if snap["running"] else None
        return snap

    def calibration(self) -> dict:
        from motion_acq.scripts.station import calibration_current

        real = self.mode == "real"
        user = str(self.settings["user"])
        if real and not user:
            return {side: {"ok": False, "detail": "보정 사용자 이름 없음"} for side in self.station.hands}
        sensecom = self.probe("gloves").get("sensecom_started_at") if real else None
        out = {}
        for side in self.station.hands:
            path = units.calibration_file(user, side, fake=not real)
            ok, detail = calibration_current(path, sensecom)
            short = str(path.relative_to(ROOT))
            out[side] = {"ok": ok, "detail": short if ok else detail.replace(str(path), short)}
        return out

    def snapshot(self) -> dict:
        st = self.station
        tcp, measured = units.tcp_calibration(st)
        all_keys = self.unit_keys() + [k for k in self.sup.keys() if k.startswith("task_")]
        running = self.sup.running()
        qv = phases.quest_view_stats(self.sup.log_tail("quest_view", 40)) if "quest_view" in running else None
        try:
            calibration = self.calibration() if st.hands else {}
        except UnitError as exc:
            calibration = {side: {"ok": False, "detail": str(exc)} for side in st.hands}
        self.left_keys()  # drops what has exited
        return {
            "station": {"name": st.name, "robot": st.robot, "has_head": st.has_head, "hands": list(st.hands),
                        "sidecars": st.sidecars, "ros_domain": st.ros_domain, "can_ports": st.can_ports,
                        "tcp": {"path": str(tcp.relative_to(ROOT)) if tcp.is_relative_to(ROOT) else str(tcp),
                                "measured": measured}},
            "mode": self.mode,
            "closing": self.closing,
            "time": time.time(),
            "units": {key: self._unit(key) for key in all_keys},
            "running": running,
            "quest_source": gates.quest_source(self),
            "quest_view": qv,
            "probes": self.probes.snapshot() if self.probes else {},
            "head": self.head_udp.get() if self.head_udp else None,
            "hands": {side: udp.get() for side, udp in self.hand_udp.items()},
            "calibration": calibration,
            "settings": self.settings,
            "space_target": self.space_target if self.space_target in running else None,
            "left_behind": self.left_behind,
            "jobs": dict(self.jobs),
            "events": list(self.events)[-12:],
        }

    def begin_shutdown(self) -> None:
        """No new starts from now on; every unit gets one SIGINT (safe pose)."""
        with self.lock:
            self.closing = True
            self.pending.clear()
        self.stop_all()

    def wait_all(self, timeout_s: float) -> list[str]:
        deadline = time.time() + timeout_s
        while self.sup.running() and time.time() < deadline:
            time.sleep(0.3)
        return self.sup.running()
