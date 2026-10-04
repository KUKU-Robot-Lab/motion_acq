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
from motion_acq.console.direction import DirectionCheck
from motion_acq.console.probes import LatestUdp, Probes, adb_path, ros_state, run
from motion_acq.console.supervisor import Supervisor, _same_process, left_behind, process_start_epoch, \
    stop_left_behind
from motion_acq.console.units import ROOT, Launch, Station, UnitError

SPACE_UNITS = ("record", "arm", "head")
ALLOWED_KEYS = {" ", "r", "q", "\x1b", "\n", "y\n", "n\n"}
CONFIRM_TTL_S = 120.0
DEFAULT_SETTINGS = {"user": "", "arm_side": "right", "scale": units.DEFAULT_SCALE,
                    "task": "teleop demonstration", "episodes": 10,
                    "webxr_arm_ok": ""}  # set once by the direction check: arms may follow the headset view
SENSECOM = ROOT / "ros_ws/install/senseglove_com/share/senseglove_com/Linux/SenseCom_Linux_Latest/SenseCom.x86_64"
HAND_NODE_VISIBLE_S = 20.0


def same_commands(a: list[Launch], b: list[Launch]) -> bool:
    return len(a) == len(b) and all(same_command(x, y) for x, y in zip(a, b, strict=True))


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
        self.direction = DirectionCheck(ROOT / units.STATIONS_DIR / f"{station.name}.yaml")
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
            return units.quest_view(st, mode)
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
            cameras = st.cameras if self.sup.is_running("quest_view") else ()
            return units.record(st, mode, str(s["arm_side"]), float(s["scale"]), streams=streams,
                                task=str(s["task"]), episodes=int(s["episodes"]), output_dir=out, cameras=cameras)
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
            launch = self.build(key)
            extra = ""
            if key == "record":
                missing = [item["name"] for item in self.record_checklist() if not item["ok"]]
                extra = (" 빠진 것: " + ", ".join(missing) + ".") if missing else " 모든 스트림이 준비됨."
            return self._gate(launch, confirm=confirm, token=token, extra=extra)

    def _approve(self, key: str, launches: list[Launch], *, title: str, summary: str, confirm: bool,
                 token: str) -> tuple[dict | None, list[Launch]]:
        """(None, the launches to run) once approved or if nothing moves; (need_confirm reply, []) otherwise.

        Approval = the token handed out with the exact commands, within CONFIRM_TTL_S, for the same
        commands (a record --output-dir timestamp may differ). The stored launches are the ones run."""
        if not any(launch.moves_robot for launch in launches):
            return None, launches
        pending = self.pending.get(key)
        approved = (pending is not None and confirm and bool(token)
                    and time.time() - pending[2] < CONFIRM_TTL_S and same_commands(pending[1], launches)
                    and secrets.compare_digest(pending[0], token))
        if pending is not None and approved:
            del self.pending[key]
            return None, pending[1]
        new_token = secrets.token_hex(8)
        self.pending[key] = (new_token, launches, time.time())
        reply = {"ok": False, "need_confirm": True, "token": new_token, "title": title,
                 "summary": ("확인한 뒤 명령이 바뀌었거나 시간이 지났다. 다시 확인할 것. " if confirm else "") + summary,
                 "command": "\n".join(launch.display() for launch in launches)}
        return reply, []

    def _gate(self, launch: Launch, *, confirm: bool, token: str, extra: str = "") -> dict:
        """Start launch, or ask for (and later honour) the operator's confirmation of its exact command."""
        reply, launches = self._approve(launch.key, [launch], title=launch.title, summary=launch.summary + extra,
                                        confirm=confirm, token=token)
        if reply is not None:
            return reply
        return self._spawn(launches[0], confirmed=launches[0].moves_robot)

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
        if kind == "quest" and arg in ("app", "view", "connect", "start"):
            fn = {"app": self._quest_app, "view": self._quest_view, "connect": self._quest_connect,
                  "start": self._quest_start}[arg]
            return self._job("quest", fn)
        if kind == "glove" and arg == "connect":
            return self._job("glove", self._glove_connect)
        if kind == "direction":
            return self._direction(arg)
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
        if kind == "hand_on":
            return self._hand_on(units.check_side(arg), confirm=confirm, token=token)
        if kind == "hand_off":
            return self.stop(f"hand_{units.check_side(arg)}")  # SIGINT: the node walks the hand open, exits
        return {"ok": False, "error": f"unknown action {name!r}"}

    # -- robot hand: node + enable in one confirmed step -------------------------------------------
    def _hand_on(self, side: str, *, confirm: bool, token: str) -> dict:
        key = f"hand_{side}"
        with self.lock:
            node_running = self.sup.is_running(key)
            if not node_running:
                problems = self.blockers(key)
                if problems:
                    return {"ok": False, "error": " / ".join(problems)}
            launches = ([] if node_running else [self.build(key)]) + [
                units.hand_enable(self.station, self.mode, side, True)]
            side_ko = "오른손" if side == "right" else "왼손"
            reply, approved = self._approve(f"hand_on_{side}", launches, title=f"로봇 {side_ko} 켜기",
                                            summary=f"{side_ko} 노드를 띄우고(필요하면) 켠다: home(펼침) 뒤 장갑을 따라간다.",
                                            confirm=confirm, token=token)
            if reply is not None:
                return reply
        return self._job(f"hand_{side}", lambda: self._hand_on_job(side, approved))

    def _hand_on_job(self, side: str, launches: list[Launch]) -> str:
        key = f"hand_{side}"
        for launch in launches[:-1]:  # the node, if it was not running
            result = self._spawn(launch, confirmed=True)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "hand node"))
        deadline = time.time() + HAND_NODE_VISIBLE_S
        while True:  # our node must be visible on the domain before the enable is sent
            if not self.sup.is_running(key):
                raise RuntimeError("손 노드가 끝났다: 로그 확인")
            ros = ros_state(self.station, self.mode)
            ros["at"] = time.time()
            problems = gates.hand_blockers(self, side, ros, starting=False)
            if not problems:
                break
            if time.time() > deadline:
                raise RuntimeError(" / ".join(problems))
            time.sleep(1.0)
        result = self._spawn(launches[-1], confirmed=launches[-1].moves_robot)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "enable"))
        if self._wait_exit(launches[-1].key, 30) != 0:
            raise RuntimeError("켜기 토픽 발행 실패: 로그 확인")
        return "켬: home(펼침) 뒤 장갑을 따라간다"

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
        rc = self._wait_exit(self._spawn_task(units.quest_task(self.station, "app")), 40)
        if rc != 0:
            raise RuntimeError(f"quest_usb.sh app 실패 (rc {rc}): 로그 확인")
        return "HandUMI 앱 연결됨"

    def _quest_view(self) -> str:
        """Head camera in the headset in one go: [연결] then [시작]."""
        self._quest_users_check()
        return f"{self._quest_connect()} / {self._quest_start()}"

    def _quest_connect(self) -> str:
        """[연결]: real = quest-view (camera + pose server) and the page in the headset; fake = mock + pattern."""
        if self.mode == "fake":
            for key in ("mock_quest", "quest_view"):
                if not self.sup.is_running(key):
                    result = self._spawn(self.build(key))
                    if not result.get("ok"):
                        raise RuntimeError(result.get("error", key))
            return "가짜 Quest + 테스트 영상"
        self._start_quest_view()
        if self._wait_exit(self._spawn_task(units.quest_task(self.station, "page")), 40) != 0:
            raise RuntimeError("헤드셋에 페이지를 못 열었다(USB·USB 디버깅 허용 확인): 로그 확인")
        return "헤드셋에 페이지가 열렸다: 헤드셋을 쓰고 [시작]"

    def _quest_start(self) -> str:
        """[시작]: enter VR on the open page from the PC (the camera shows in the headset)."""
        if self.mode == "fake":
            return "fake: 시작할 것 없음"
        if not self.sup.is_running("quest_view"):
            raise RuntimeError("먼저 [연결]")
        if self._wait_exit(self._spawn_task(units.quest_task(self.station, "vr")), 40) != 0:
            raise RuntimeError("VR 이 시작되지 않았다: 헤드셋을 쓰고(깨우고) 다시 [시작]")
        return "VR 시작: 헤드셋에 카메라 영상"

    def _start_quest_view(self) -> None:
        with self.lock:
            if self.sup.is_running("quest_view"):
                return  # reopening the page (e.g. after a USB re-plug) is fine while head / arms run
            self._quest_users_check()  # they hold the HandUMI app stream that quest-view replaces
            problems = list(self.blockers("quest_view"))
            if gates.quest_source(self) == "external":
                problems.append(gates.external_quest(self))
            if problems:
                raise RuntimeError(" / ".join(problems))
            run([adb_path(), "forward", "--remove", "tcp:65432"], timeout=5)
            result = self._spawn(units.quest_view(self.station, self.mode))
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "quest-view"))
        deadline = time.time() + 15
        while not any("page http://localhost" in line for line in self.sup.log_tail("quest_view", 50)):
            if not self.sup.is_running("quest_view") or time.time() > deadline:
                raise RuntimeError("quest-view 가 뜨지 않았다: 로그 확인")
            time.sleep(0.3)

    def _spawn_task(self, launch: Launch) -> str:
        result = self._spawn(launch)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", launch.key))
        return launch.key

    def _glove_connect(self) -> str:
        """[연결]: SenseCom + both gloves over BLE, then the glove driver as soon as they are connected."""
        if self.mode != "real":
            raise RuntimeError("장갑은 실기 모드에서만 연결한다")
        if not SENSECOM.exists():
            raise RuntimeError(f"SenseCom 미설치: 이 PC 에서 scripts/ros_ws_setup.sh --full 이 필요하다 ({SENSECOM.name})")
        if self._wait_exit(self._spawn_task(units.glove_task(self.station, self.mode, "up")), 90) != 0:
            raise RuntimeError("장갑이 연결되지 않았다(전원·LED, 다른 PC 의 SenseCom 확인): 로그 확인")
        if self.probe("gloves").get("driver_pids"):
            return "장갑 연결됨, 드라이버는 이미 실행 중"
        if self._wait_exit(self._spawn_task(units.glove_task(self.station, self.mode, "driver")), 40) != 0:
            raise RuntimeError("장갑 드라이버가 뜨지 않았다: 로그 확인")
        return "장갑 연결 + 드라이버 실행"

    def _direction(self, arg: str) -> dict:
        if arg == "on":
            if gates.quest_source(self) not in ("view", "mock"):
                return {"ok": False, "error": "방향 확인은 헤드셋 영상 모드(또는 가짜 Quest)에서: 먼저 [연결]·[시작]"}
            self.direction.start()
        elif arg == "off":
            self.direction.stop()
        elif arg == "rebase":
            self.direction.rebase()
        elif arg == "ok":
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            with self.lock:
                self.settings = {**self.settings, "webxr_arm_ok": stamp}
                self._settings_path().write_text(json.dumps(self.settings, ensure_ascii=False, indent=1))
            self.direction.stop()
            self.event("info", "방향 확인 완료: 헤드셋 영상 모드로 팔을 움직일 수 있다")
        elif arg == "reset":
            with self.lock:
                self.settings = {**self.settings, "webxr_arm_ok": ""}
                self._settings_path().write_text(json.dumps(self.settings, ensure_ascii=False, indent=1))
        else:
            return {"ok": False, "error": f"unknown direction action {arg!r}"}
        self.intent("direction", what=arg)
        return {"ok": True}

    # -- recording: everything at once ------------------------------------------------------------------
    def record_checklist(self) -> list[dict]:
        """What a recording started now would contain; anything not ok is missing from the dataset."""
        st, running = self.station, set(self.sup.running())
        source = gates.quest_source(self)
        items = [{"name": "Quest 자세", "ok": source in ("view", "app", "mock"), "detail": source or "없음"},
                 {"name": "로봇 팔", "ok": "arm" not in running, "detail": "팔 원격조작을 먼저 정지" if "arm" in running
                  else {"right": "오른팔", "left": "왼팔", "both": "양팔"}[str(self.settings["arm_side"])]}]
        if st.cameras:
            items.append({"name": "카메라 영상", "ok": "quest_view" in running,
                          "detail": "quest-view 영상" if "quest_view" in running else "Quest [연결] 필요"})
        if st.has_head:
            items.append({"name": "목", "ok": "head" in running, "detail": "실행 중" if "head" in running else "꺼짐"})
        for side in st.hands:
            rec = self.hand_udp[side].get() if side in self.hand_udp else None
            on = f"hand_{side}" in running and bool(rec) and rec.get("mode") == "enabled"
            items.append({"name": f"로봇 {'오른손' if side == 'right' else '왼손'}", "ok": on,
                          "detail": "켜짐" if on else "꺼짐"})
        return items

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
            "record_checklist": self.record_checklist(),
            "direction": self.direction.snapshot(),
            "sensecom_installed": SENSECOM.exists(),
            "jobs": dict(self.jobs),
            "events": list(self.events)[-12:],
        }

    def begin_shutdown(self) -> None:
        """No new starts from now on; every unit gets one SIGINT (safe pose)."""
        with self.lock:
            self.closing = True
            self.pending.clear()
        self.direction.stop()
        self.stop_all()

    def wait_all(self, timeout_s: float) -> list[str]:
        deadline = time.time() + timeout_s
        while self.sup.running() and time.time() < deadline:
            time.sleep(0.3)
        return self.sup.running()
