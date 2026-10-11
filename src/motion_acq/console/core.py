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
import string
import threading
import time
from pathlib import Path

from motion_acq.console import gates, phases, units
from motion_acq.console.direction import DirectionCheck
from motion_acq.console.probes import LatestUdp, Probes, adb_path, ros_state, run
from motion_acq.console.supervisor import Supervisor, _same_process, left_behind, stop_left_behind
from motion_acq.console.units import ROOT, Launch, Station, UnitError

SPACE_UNITS = ("record", "arm", "arm_right", "arm_left", "head")
ALLOWED_KEYS = {" ", "r", "q", "h", "\x1b", "\n", "y\n", "n\n"}
CONFIRM_TTL_S = 120.0
DEFAULT_SETTINGS = {"user": "", "arm_side": "right", "scale": units.DEFAULT_SCALE,
                    "task": "teleop demonstration", "episodes": 10,
                    "webxr_arm_ok": ""}  # set once by the direction check: arms may follow the headset view
SENSECOM = ROOT / "ros_ws/install/senseglove_com/share/senseglove_com/Linux/SenseCom_Linux_Latest/SenseCom.x86_64"
HAND_NODE_VISIBLE_S = 20.0
HAND_LINK_NOTE = (" 손: 노드가 꺼진 채 먼저 뜨고, 팔이 손을 주먹으로 쥔 뒤 차렷 -> 홈. 홈에서 손이 펴지면 1 초 동안 "
                  "손을 편 채 멈춰 있을 것(기준 자세), 그 뒤 장갑을 따라간다. 정지 때는 손을 주먹으로 쥔 뒤 차렷.")
# one arm alone (10.11 user, option B): its hand needs the driver, not the glove, and stays closed at home
ONE_ARM_HAND_NOTE = (" 손: 드라이버·노드를 띄워 주먹으로 쥔다(장갑 필요 없음). 팔이 차렷 -> 홈에 와도 손은 주먹 그대로: "
                     "펴려면 손 칸 [홈] / [따라가기]. [차렷] 때도 손을 주먹으로 쥔 뒤 차렷.")
HAND_DRIVER_UP_S = 30.0  # EtherCAT: OP and the first /hand_<side>/angle_actual
# arm / record: everything at once (the arm_side setting); arm_right / arm_left: one arm alone (10.11 user)
ARM_UNITS = ("arm", "record", "arm_right", "arm_left")


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
        if key in ("arm_right", "arm_left"):
            return units.arm(st, mode, key[4:], float(s["scale"]), key=key)
        if key == "record":
            streams = units.record_streams(st, set(self.sup.running()))
            out = ROOT / "outputs" / f"{st.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            cameras = st.cameras if self.sup.is_running("quest_view") else ()
            return units.record(st, mode, str(s["arm_side"]), float(s["scale"]), streams=streams,
                                task=str(s["task"]), episodes=int(s["episodes"]), output_dir=out, cameras=cameras)
        if key.startswith("hand_"):
            return units.hand(st, mode, key[5:], str(s["user"]), self.session_dir / "hand")
        if key.startswith("ecat_"):
            return units.hand_driver(st, mode, key[5:])
        if key.startswith("calib_"):
            return units.calibrate(st, mode, key[6:], str(s["user"]))
        if key.startswith("rezero_"):
            return units.calibrate(st, mode, key[7:], str(s["user"]), rezero=True)
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
            if key in ARM_UNITS and self._arm_hand_sides(key):
                return self._start_arm_with_hands(launch, extra, confirm=confirm, token=token)
            return self._gate(launch, confirm=confirm, token=token, extra=extra)

    # -- arm start with the RH56F1 hands (10.06 user: fist -> rest -> home -> fingers) ----------------
    def arm_sides(self, key: str) -> tuple[str, ...]:
        """The arms a unit moves: arm_right / arm_left one, arm / record the arm_side setting."""
        if key in ("arm_right", "arm_left"):
            return (key[4:],)
        side = str(self.settings["arm_side"])
        return ("right", "left") if side == "both" else (side,)

    def arms_on(self, side: str) -> list[str]:
        """Running arm units that move this side's arm."""
        return [k for k in ARM_UNITS if self.sup.is_running(k) and side in self.arm_sides(k)]

    def _arm_hand_sides(self, key: str) -> tuple[str, ...]:
        return tuple(s for s in self.arm_sides(key) if s in self.station.hand_link and s in self.station.hands)

    def _start_arm_with_hands(self, arm: Launch, extra: str, *, confirm: bool, token: str) -> dict:
        """[홈] of an arm / [전체 시작] brings up the hand driver and node of each arm side first (disabled).
        The arm then closes the hands to the fist (hand_link), plays rest -> home, and at home the hands
        open. All at once (arm / record) the hands are set back to auto first, so they follow the gloves
        when the arms follow; one arm alone leaves its hand's [홈] / [따라가기] as the operator set it."""
        sides = self._arm_hand_sides(arm.key)
        one_arm = arm.key not in ("arm", "record")
        note = ONE_ARM_HAND_NOTE if one_arm else HAND_LINK_NOTE
        problems, launches, commands = [], [], []
        for side in sides:
            if self.sup.is_running(f"hand_{side}"):
                rec = self.hand_udp[side].get() if side in self.hand_udp else None
                if one_arm and (rec or {}).get("manual") != "fist":
                    commands.append(units.hand_command(self.station, self.mode, side, "fist"))
                elif not one_arm and (rec or {}).get("manual"):
                    commands.append(units.hand_command(self.station, self.mode, side, "auto"))
                continue
            side_ko = "오른손" if side == "right" else "왼손"
            problems += [f"{side_ko}: {p}" for p in self._hand_precheck(side, need_glove=not one_arm)]
            if self.mode == "real" and not self._hand_driver_present(side):
                launches.append(units.hand_driver(self.station, self.mode, side))
            launches.append(self.build(f"hand_{side}"))
            if one_arm:
                commands.append(units.hand_command(self.station, self.mode, side, "fist"))
        if problems:
            return {"ok": False, "error": " / ".join(problems)}
        launches += commands
        if not launches:
            return self._gate(arm, confirm=confirm, token=token, extra=extra + note)
        launches.append(arm)
        reply, approved = self._approve(arm.key, launches, title=arm.title,
                                        summary=arm.summary + note + extra, confirm=confirm, token=token)
        if reply is not None:
            return reply
        return self._job(arm.key, lambda: self._arm_with_hands_job(sides, approved))

    def _arm_with_hands_job(self, sides: tuple[str, ...], launches: list[Launch]) -> str:
        commands = [launch for launch in launches[:-1] if launch.key.startswith("task_hand_")]
        for launch in launches[:-1]:
            if launch in commands:  # after the nodes are up: fist (one arm) / auto (all at once)
                continue
            side = launch.key.split("_", 1)[1]
            if launch.key.startswith("ecat_"):
                self._spawn_task_checked(launch)
                self._wait_hand(side, starting=True, need_node=False, timeout_s=HAND_DRIVER_UP_S)
            else:
                if self.mode == "real":
                    self._wait_hand(side, starting=True, need_node=False, timeout_s=5.0)
                self._spawn_task_checked(launch)
        for side in sides:
            self._wait_hand(side, starting=False, need_node=True, timeout_s=HAND_NODE_VISIBLE_S)
        for launch in commands:
            if self._wait_exit(self._spawn_task(launch), 30) != 0:
                raise RuntimeError(f"{launch.title} 실패: 로그 확인")
        self._spawn_task_checked(launches[-1])
        if launches[-1].key in ("arm", "record"):
            return "팔 시작: 손 주먹 -> 차렷 -> 홈 -> 손 펴고 기다림 (Space: 팔과 손이 따라감)"
        return "팔 시작: 손 주먹 -> 차렷 -> 홈 (손은 주먹 그대로, Space: 팔만 따라감)"

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
        if key in ARM_UNITS:
            # the stored rest path needs the hand at the fist, confirmed by its node (hand_link.py):
            # without the node the arm would refuse its path and hold at home
            off = [s for s in self._arm_hand_sides(key) if not self.sup.is_running(f"hand_{s}")]
            if off:
                names = ", ".join(f"로봇 {'오른' if s == 'right' else '왼'}손" for s in off)
                return {"ok": False, "error": f"{names} 노드가 꺼져 있다: 차렷 경로는 손이 주먹이어야 하므로 "
                                              f"{names} [주먹]을 먼저 누를 것 (모두 정지는 그대로 동작)"}
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
        if text == " " and key in gates.FOLLOW_KEYS and self.mode == "real":
            problems = gates.follow_blockers(self, gates.quest_source(self))
            if problems:  # the arm stays at home; it would only log the ignored Space
                return {"ok": False, "error": "따라가기 불가: " + " / ".join(problems)}
        try:
            self.sup.send_key(key, text)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        self.intent("key", key=key, text=repr(text))
        return {"ok": True}

    def stop_all(self) -> dict:
        """One SIGINT each, in order: arms, head, Quest, gloves first; the robot hands only once the
        arms are back at rest (the stored paths assume the hand as it is, not opened on the way);
        the hand drivers last, after the hand nodes have opened the hands."""
        running = self.sup.running()
        first = [k for k in running if not (k.startswith("hand_") or k.startswith("ecat_"))]
        stopped = [key for key in first if self.sup.stop(key)]
        later = [k for k in running if k.startswith("hand_") or k.startswith("ecat_")]
        if later:
            threading.Thread(target=self._stop_hands_after_arms, daemon=True, name="console-stop-hands").start()
        self.intent("stop_all", keys=stopped, later=later)
        self.event("warn", "모두 정지 요청: " + (", ".join(stopped + later) or "없음")
                   + (" (손은 팔이 차렷에 간 뒤)" if later else ""))
        return {"ok": True, "stopped": stopped, "later": later}

    def _stop_hands_after_arms(self) -> None:
        deadline = time.time() + 180
        while any(self.sup.is_running(k) for k in ARM_UNITS) and time.time() < deadline:
            time.sleep(0.3)
        hands = [k for k in self.sup.running() if k.startswith("hand_")]
        for key in hands:
            self.sup.stop(key)
        deadline = time.time() + 30
        while any(self.sup.is_running(k) for k in hands) and time.time() < deadline:
            time.sleep(0.3)
        for key in [k for k in self.sup.running() if k.startswith("ecat_")]:
            self.sup.stop(key)

    def set_space_target(self, key: str | None) -> dict:
        if key is not None and key not in SPACE_UNITS:
            return {"ok": False, "error": "Space 대상이 될 수 없다"}
        self.space_target = key
        return {"ok": True}

    # -- tasks and sequences --------------------------------------------------------------------
    def action(self, name: str, *, confirm: bool = False, token: str = "") -> dict:
        kind, _, arg = name.partition(":")
        if kind == "quest" and arg in ("app", "view", "connect", "start", "offhead"):
            fn = {"app": self._quest_app, "view": self._quest_view, "connect": self._quest_connect,
                  "start": self._quest_start, "offhead": self._quest_offhead}[arg]
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
        if kind == "hand":
            side, _, what = arg.partition(":")
            if what not in ("home", "follow", "fist"):
                return {"ok": False, "error": f"unknown hand command {what!r}"}
            return self._hand_cmd(units.check_side(side), what, confirm=confirm, token=token)
        if kind == "hand_off":
            # 10.11 user: each hand turns off on its own, arm on or off. SIGINT: the node closes the hand to
            # the fist and exits; the arm keeps its pose. Its [차렷] then waits for the hand ([주먹] / [홈]).
            return self.stop(f"hand_{units.check_side(arg)}")
        if kind == "driver_off":
            side = units.check_side(arg)
            if self.sup.is_running(f"hand_{side}"):
                return {"ok": False, "error": "손을 먼저 끌 것([끄기]): 드라이버가 먼저 내려가면 손이 그 자리에서 멈춘다"}
            return self.stop(f"ecat_{side}")
        return {"ok": False, "error": f"unknown action {name!r}"}

    # -- robot hand: driver + node + enable in one confirmed step ------------------------------------
    def _hand_driver_present(self, side: str) -> bool:
        ros = gates.fresh_ros(self, self.probe("ros"))
        return bool(self.sup.is_running(f"ecat_{side}") or gates.driver_running_elsewhere(self, side)
                    or (ros and ((ros.get("hands") or {}).get(side) or {}).get("driver")))

    def _hand_precheck(self, side: str, *, need_glove: bool) -> list[str]:
        """What stops a hand command before anything starts (the driver may still be off: it starts first).
        The glove only for following (10.11 user: the arms and [홈] / [주먹] go without the Nova)."""
        if self.closing:
            return ["콘솔이 종료 중이다"]
        out = []
        cal = self.calibration().get(side) or {}
        if not cal.get("ok"):
            out.append(f"장갑 보정이 필요하다: {cal.get('detail', '?')} (장갑 > 보정)")
        if self.mode == "real":
            ros = gates.fresh_ros(self, self.probe("ros"))
            if ros is None:
                out.append("ROS 상태 확인 중: 잠시 뒤 다시")
            else:
                if need_glove and not (ros.get("glove_topics") or {}).get(side):
                    out.append("장갑 토픽 없음: 장갑 [연결] 먼저")
                nodes = ((ros.get("hands") or {}).get(side) or {}).get("angle_set_publisher_nodes") or []
                foreign = [n for n in nodes if n != f"motion_acq_hand_{side}"]
                if foreign:
                    out.append(f"/hand_{side}/angle_set 을 다른 노드({', '.join(foreign)})가 발행 중: 그쪽 손 제어를 먼저 끌 것")
        return out

    def _hand_cmd(self, side: str, kind: str, *, confirm: bool, token: str) -> dict:
        """[홈] / [따라가기] of one hand (10.11 user: each hand on its own): driver and node if missing, then
        the command. Also clears a latched FAULT (HandController.command)."""
        key = f"hand_{side}"
        with self.lock:
            problems = self._hand_precheck(side, need_glove=kind == "follow")
            if problems:
                return {"ok": False, "error": " / ".join(problems)}
            launches = []
            if self.mode == "real" and not self._hand_driver_present(side):
                launches.append(units.hand_driver(self.station, self.mode, side))
            if not self.sup.is_running(key):
                launches.append(self.build(key))
            command = units.hand_command(self.station, self.mode, side, kind)
            launches.append(command)
            steps = " → ".join(["드라이버" if item.key.startswith("ecat_") else "노드" if item.key == key
                                else units.HAND_COMMANDS[kind] for item in launches])
            reply, approved = self._approve(f"hand_{kind}_{side}", launches, title=f"로봇 {command.title}",
                                            summary=f"{steps}. {command.summary}", confirm=confirm, token=token)
            if reply is not None:
                return reply
        return self._job(key, lambda: self._hand_cmd_job(side, approved, need_glove=kind == "follow"))

    def _wait_hand(self, side: str, *, starting: bool, need_node: bool, timeout_s: float,
                   need_glove: bool = False) -> None:
        deadline = time.time() + timeout_s
        while True:
            if need_node and not self.sup.is_running(f"hand_{side}"):
                raise RuntimeError("손 노드가 끝났다: 로그 확인")
            if self.mode == "real" and f"ecat_{side}" in self.sup.children and not self.sup.is_running(f"ecat_{side}") \
                    and not gates.driver_running_elsewhere(self, side):
                raise RuntimeError("손 드라이버가 끝났다: 손 전원·랜선, 드라이버 로그 확인")
            ros = ros_state(self.station, self.mode)
            ros["at"] = time.time()
            problems = gates.hand_blockers(self, side, ros, starting=starting, need_glove=need_glove)
            if not problems:
                return
            if time.time() > deadline:
                raise RuntimeError(" / ".join(problems))
            time.sleep(1.0)

    def _hand_cmd_job(self, side: str, launches: list[Launch], *, need_glove: bool = False) -> str:
        command = launches[-1]
        for launch in launches[:-1]:
            if launch.key.startswith("ecat_"):
                self._spawn_task_checked(launch)
                self._wait_hand(side, starting=True, need_node=False, timeout_s=HAND_DRIVER_UP_S)
            else:  # the node: real = the driver is up and nobody else commands this hand
                if self.mode == "real":  # (fake: the node's launch brings its own fake driver)
                    self._wait_hand(side, starting=True, need_node=False, timeout_s=5.0)
                self._spawn_task_checked(launch)
        self._wait_hand(side, starting=False, need_node=True, timeout_s=HAND_NODE_VISIBLE_S, need_glove=need_glove)
        self._spawn_task_checked(command)
        if self._wait_exit(command.key, 30) != 0:
            raise RuntimeError(f"{command.title} 토픽 발행 실패: 로그 확인")
        return command.summary

    def _spawn_task_checked(self, launch: Launch) -> None:
        result = self._spawn(launch, confirmed=launch.moves_robot)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", launch.key))

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
        """Switching the Quest source (app <-> headset view) cuts the poses: not under a following arm.
        An arm waiting at home does not use them (10.11 user: arms first, the Quest later)."""
        users = [k for k in gates.QUEST_PRODUCERS if self.sup.is_running(k)]
        users += [k for k in gates.FOLLOW_KEYS if self.sup.is_running(k) and self._arm_following(k)]
        if users:
            raise RuntimeError(f"Quest 를 쓰는 중({', '.join(users)}): 머리/녹화는 정지, 팔은 [홈] 먼저")

    def _arm_following(self, key: str) -> bool:
        ph = phases.phase(key, self.sup.log_tail(key, 120))
        return ph is not None and ph[1] in phases.ARM_FOLLOWING

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

    def _quest_offhead(self) -> str:
        """[안 쓰고 시작] (10.11 user): proximity sensor off + wake, then [연결] if needed and [시작].

        For setting up without wearing the headset; [근접 센서 원래대로] in 자세히 undoes it."""
        if self.mode == "fake":
            return "fake: 헤드셋 없음"
        if self._wait_exit(self._spawn_task(units.quest_task(self.station, "unworn")), 20) != 0:
            raise RuntimeError("헤드셋이 깨지 않았다: 전원 버튼을 한 번 누르고 다시")
        self._quest_users_check()
        steps = ["근접 센서 끔"]
        if not self.sup.is_running("quest_view"):
            steps.append(self._quest_connect())
        steps.append(self._quest_start())
        return " / ".join(steps)

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
            missing = self._missing_glove_topics()
            if not missing:
                return "장갑 연결됨, 드라이버는 이미 실행 중"
            # e.g. one glove's ros2_control_node died (libsgcore aborts on a sensor string it cannot parse)
            self.event("warn", f"장갑 드라이버가 떠 있지만 {', '.join(missing)} 데이터가 없다: 드라이버를 다시 시작")
            if self._wait_exit(self._spawn_task(units.glove_task(self.station, self.mode, "stop")), 30) != 0:
                raise RuntimeError("장갑 드라이버를 내리지 못했다: 로그 확인")
        if self._wait_exit(self._spawn_task(units.glove_task(self.station, self.mode, "driver")), 40) != 0:
            raise RuntimeError("장갑 드라이버가 뜨지 않았다: 로그 확인")
        missing = self._missing_glove_topics()
        if missing:
            raise RuntimeError(f"장갑 드라이버는 떴지만 {', '.join(missing)} 데이터가 없다: 드라이버 로그 확인")
        return "장갑 연결 + 드라이버 실행"

    def _missing_glove_topics(self) -> list[str]:
        """Glove sides whose senseglove_states is not on the domain right now (read directly)."""
        ros = ros_state(self.station, self.mode)
        if "error" in ros:
            return list(self.station.hands)
        topics = ros.get("glove_topics") or {}
        return [side for side in self.station.hands if not topics.get(side)]

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
        arms = [k for k in ("arm", "arm_right", "arm_left") if k in running]
        items = [{"name": "Quest 자세", "ok": source in ("view", "app", "mock"), "detail": source or "없음"},
                 {"name": "로봇 팔", "ok": not arms, "detail": "팔 원격조작을 먼저 정지" if arms
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
        """nova2.sh stop: the driver's whole process group (a background ros2 launch ignores SIGINT)."""
        if any(self.sup.is_running(f"hand_{side}") for side in self.station.hands):
            return {"ok": False, "error": "로봇 손을 먼저 끌 것: 장갑이 멈추면 손이 그 자리에서 멈춘다"}
        return self._spawn(units.glove_task(self.station, self.mode, "stop"))

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
        keys += [f"arm_{side}" for side in ("right", "left") if side in self.station.can_ports]
        for side in self.station.hands:
            keys += [f"hand_{side}", f"calib_{side}", f"rezero_{side}"] + ([f"ecat_{side}"] if self.mode == "real" else [])
        return keys

    def _unit(self, key: str) -> dict:
        snap = self.sup.snapshot(key)
        tail = self.sup.log_tail(key, 120)
        snap["phase"] = phases.phase(key, tail) if snap["running"] or snap["rc"] is not None else None
        snap["prompt"] = phases.prompt(tail, snap.get("partial", "")) if snap["running"] else None
        snap["error"] = phases.error_line(tail) if snap["rc"] not in (None, 0) else None
        return snap

    def calibration(self) -> dict:
        from motion_acq.scripts.station import calibration_current, calibration_stale

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
            stale = ok and calibration_stale(path, sensecom)
            out[side] = {"ok": ok, "stale": stale,
                         "detail": f"{short} (SenseCom 을 다시 켠 뒤: 손이 어긋나면 [편 손 맞춤] 2초)"
                         if stale else short if ok else detail.replace(str(path), short)}
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
