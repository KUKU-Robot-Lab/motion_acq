"""What the console runs on a station: argv, env and safety class of each unit.

Pure builders (no processes started here), so the exact command the operator
approves is the one that runs. Units are long-running programs under the
supervisor; tasks are short commands (quest_usb.sh, nova2.sh, ros2 topic pub)
run the same way and shown with their exit code.

Station facts come from configs/stations/<station>.yaml: the robot, the CAN
repair policy, the head section, recording.sidecars (which hands exist) and
hands.ros_domain_id (the s2r real domain the RH56F1 EtherCAT driver runs on).
motion_acq never starts the hand driver: the s2r console brings it up.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from motion_acq.config import STATION_ENV, STATIONS_DIR, load_rig_config

ROOT = Path(__file__).resolve().parents[3]
VENV_BIN = Path(sys.executable).parent
MACQ = str(VENV_BIN / "macq")
SIDES = ("right", "left")
ARM_SIDES = ("right", "left", "both")
FAKE_DOMAIN = "177"

# UDP ports the console listens on (copies of the records the recorder gets).
CONSOLE_HEAD_PORT = 47131
CONSOLE_HAND_PORTS = {"right": 47141, "left": 47142}
QUEST_VIEW_HEAD_PORT = 47121  # motion_acq.quest_view.server.HEAD_STATUS_PORT

IDENTITY_TCP = ROOT / "configs" / "calibration" / "controller_tcp" / "identity.yaml"
SCALE_RANGE = (0.1, 1.7)
DEFAULT_SCALE = 0.5  # the arm4090 first real runs (10.04) used 0.5


class UnitError(ValueError):
    """A unit cannot be built for this station or these options (message for the operator)."""


@dataclass(frozen=True)
class Station:
    name: str
    rig: dict

    @classmethod
    def load(cls, name: str) -> Station:
        path = ROOT / STATIONS_DIR / f"{name}.yaml"
        if not name or not path.exists():
            known = sorted(p.stem for p in (ROOT / STATIONS_DIR).glob("*.yaml"))
            raise UnitError(f"unknown station {name!r}; set {STATION_ENV} to one of {', '.join(known)}")
        return cls(name, load_rig_config(path))

    @property
    def robot(self) -> str:
        return str((self.rig.get("recording") or {}).get("robot") or "openarmv1")

    @property
    def sidecars(self) -> dict[str, int]:
        raw = (self.rig.get("recording") or {}).get("sidecars") or {}
        return {str(k): int(v) for k, v in raw.items()}

    @property
    def has_head(self) -> bool:
        return isinstance(self.rig.get("head"), dict)

    @property
    def hands(self) -> tuple[str, ...]:
        return tuple(side for side in SIDES if f"hand_{side}" in self.sidecars)

    @property
    def ros_domain(self) -> str | None:
        value = (self.rig.get("hands") or {}).get("ros_domain_id")
        return None if value is None else str(int(value))

    @property
    def cameras(self) -> tuple[str, ...]:
        """Cameras the recorder captures (arm4090: head, frames from quest-view)."""
        return tuple(str(name) for name in (self.rig.get("cameras") or {}))

    @property
    def hand_driver(self) -> Path | None:
        """The RH56F1 EtherCAT driver script (s2r's rh56f1_driver.py), if the rig names one."""
        value = (self.rig.get("hands") or {}).get("driver")
        return None if not value else Path(str(value)).expanduser()

    @property
    def can_auto_repair(self) -> bool:
        robot = (self.rig.get("robots") or {}).get(self.robot) or {}
        return bool((robot.get("can") or {}).get("auto_repair", True))

    @property
    def can_ports(self) -> dict[str, str]:
        can = (((self.rig.get("robots") or {}).get(self.robot) or {}).get("can") or {})
        return {side: str(can[f"{side}_port"]) for side in SIDES if can.get(f"{side}_port")}


@dataclass(frozen=True)
class Launch:
    key: str
    title: str                      # Korean label for the operator
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    pty: bool = False               # reads keys (Space, R, Q, Enter)
    moves_robot: bool = False       # real mode: the operator confirms the exact argv first
    summary: str = ""               # what moves, in Korean
    stop_grace_s: float = 10.0      # a graceful stop (safe pose) may take this long

    def display(self) -> str:
        env = " ".join(f"{k}={v}" for k, v in sorted(self.env.items()))
        return (env + " " if env else "") + " ".join(self.argv)


def _real(mode: str) -> bool:
    if mode not in ("real", "fake"):
        raise UnitError(f"mode must be real or fake, not {mode!r}")
    return mode == "real"


def _base_env(station: Station) -> dict[str, str]:
    return {STATION_ENV: station.name}


def _ros_env(station: Station, mode: str) -> dict[str, str]:
    env = _base_env(station)
    if _real(mode):
        if station.ros_domain is None:
            raise UnitError(f"{station.name}: hands.ros_domain_id is not set in the station rig")
        env["ROS_DOMAIN_ID"] = station.ros_domain
    else:
        env.update(ROS_DOMAIN_ID=FAKE_DOMAIN, ROS_LOCALHOST_ONLY="1")
    return env


def ros_argv(command: str) -> tuple[str, ...]:
    """bash -c that sources ROS 2, robot_control (rh56f1_interfaces) and this repo's ros_ws."""
    from motion_acq.scripts.station import _ros_command

    try:
        return tuple(_ros_command(command, fake=False))
    except SystemExit as exc:
        raise UnitError(str(exc)) from exc


def check_side(side: str, allowed: tuple[str, ...] = SIDES) -> str:
    if side not in allowed:
        raise UnitError(f"side must be one of {', '.join(allowed)}, not {side!r}")
    return side


def check_scale(scale: float) -> float:
    lo, hi = SCALE_RANGE
    if not lo <= float(scale) <= hi:
        raise UnitError(f"translation scale must be within {lo}..{hi}, not {scale}")
    return round(float(scale), 2)


def tcp_calibration(station: Station) -> tuple[Path, bool]:
    """(path, measured): the station's controller->TCP calibration, else identity."""
    from motion_acq.calibration.control_tcp import calibration_path_for_robot_device

    try:
        path, _ = calibration_path_for_robot_device(station.robot, "meta")
    except Exception:  # noqa: BLE001 - an unknown robot config falls back like a missing file
        return IDENTITY_TCP, False
    path = path if path.is_absolute() else ROOT / path
    return (path, True) if path.exists() else (IDENTITY_TCP, False)


# -- Quest ------------------------------------------------------------------------

def quest_view(station: Station, mode: str = "real") -> Launch:
    """Real: head camera + pose server for the headset page. Fake: a moving test pattern only
    (the mock sender keeps TCP 65432), so a fake recording still has video."""
    argv = (MACQ, "quest-view") + (() if _real(mode) else ("--test-pattern", "--no-pose-server"))
    return Launch("quest_view", "헤드셋 영상 서버", argv, _base_env(station), stop_grace_s=5.0)


def mock_quest(station: Station) -> Launch:
    argv = (str(VENV_BIN / "python"), "-m", "motion_acq.tracking.mock_quest_sender",
            "--hmd-yaw-amp-deg", "20", "--hmd-pitch-amp-deg", "8")
    return Launch("mock_quest", "가짜 Quest", argv, _base_env(station), stop_grace_s=3.0)


# -- head ---------------------------------------------------------------------------

def head(station: Station, mode: str) -> Launch:
    if not station.has_head:
        raise UnitError(f"{station.name} has no head")
    real = _real(mode)
    targets = [QUEST_VIEW_HEAD_PORT, CONSOLE_HEAD_PORT]
    if "head" in station.sidecars:
        targets.append(station.sidecars["head"])
    argv = [MACQ, "head", "--backend", "real" if real else "fake", "--unlock", "key"]
    for port in targets:
        argv += ["--udp-target", f"127.0.0.1:{port}"]
    return Launch("head", "머리", tuple(argv), _base_env(station), pty=True, moves_robot=real,
                  summary="머리가 home 으로 간 뒤 잠긴 채 기다린다. Space 로 HMD 따라가기/잠금(잠그면 home 복귀).",
                  stop_grace_s=20.0)


# -- arms -------------------------------------------------------------------------------

def _arm_common(station: Station, mode: str, side: str, scale: float) -> list[str]:
    real = _real(mode)
    side = check_side(side, ARM_SIDES)
    tcp, _ = tcp_calibration(station)
    argv = ["--device", "meta", "--robot", station.robot, "--side", side,
            "--translation-scale", f"{check_scale(scale):g}", "--skip-feetech", "--space-start",
            "--skip-cameras", "--no-rerun", "--controller-tcp-calibration", str(tcp)]
    if real and not station.can_auto_repair:
        argv.append("--skip-can-repair")  # s2r owns the CAN links on this station
    if not real:
        argv += ["--fake-robot", "--no-sounds"]
    return argv


SIDE_KO = {"right": "오른팔", "left": "왼팔", "both": "양팔"}


def arm(station: Station, mode: str, side: str, scale: float = DEFAULT_SCALE) -> Launch:
    argv = (MACQ, "teleop-real", *_arm_common(station, mode, side, scale))
    return Launch("arm", f"팔 원격조작 ({SIDE_KO[side]})", argv, _base_env(station), pty=True,
                  moves_robot=_real(mode),
                  summary=f"{SIDE_KO[side]}: 차렷이 벗어나 있으면 먼저 0.05 rad/s 로 기준 차렷에 맞추고, "
                          "저장 경로로 home 에 간 뒤 Space 를 기다린다. Space 순간 착용자가 바라보는 방향이 로봇 앞(+x)이 된다. "
                          "정지하면 home → 차렷으로 돌아간 뒤 모터를 끈다.",
                  stop_grace_s=120.0)


def record_streams(station: Station, running: set[str]) -> dict[str, int]:
    """Sidecars the recorder will require: only streams whose producer runs now."""
    streams = {}
    for name, port in station.sidecars.items():
        producer = "head" if name == "head" else name
        if producer in running:
            streams[name] = port
    return streams


def record(station: Station, mode: str, side: str, scale: float, *, streams: dict[str, int],
           task: str, episodes: int, output_dir: Path, cameras: tuple[str, ...] = ()) -> Launch:
    if not task.strip():
        raise UnitError("task description is empty")
    if not 1 <= int(episodes) <= 500:
        raise UnitError("episodes must be within 1..500")
    argv = [MACQ, "teleop-record", *_arm_common(station, mode, side, scale),
            "--task", task.strip(), "--num-episodes", str(int(episodes)), "--output-dir", str(output_dir)]
    if cameras:
        argv = [a for a in argv if a != "--skip-cameras"] + ["--cameras", ",".join(cameras)]
    if streams:
        for name, port in streams.items():
            argv += ["--sidecar", f"{name}={port}"]
    else:
        argv.append("--no-sidecars")
    names = ", ".join([*streams, *(f"카메라 {c}" for c in cameras)]) or "팔만"
    return Launch("record", f"녹화 ({SIDE_KO[side]}, {names})", tuple(argv), _base_env(station), pty=True,
                  moves_robot=_real(mode),
                  summary=f"{SIDE_KO[side]}: 차렷 → home 이동 후 Space 로 에피소드 시작/저장, R 다시, Q 마침.",
                  stop_grace_s=120.0)


# -- hands (Nova 2 -> RH56F1 over the s2r EtherCAT driver) ------------------------------------------

def calibration_file(user: str, side: str, *, fake: bool = False) -> Path:
    """Real: configs/hands/calibration/<user>_<side>.yaml; fake: the fake_hand_check.sh file (as macq station)."""
    if fake:
        return ROOT / "logs" / "hand" / f"fake_calibration_{check_side(side)}.yaml"
    if not user or not user.replace("_", "").replace("-", "").isalnum():
        raise UnitError("calibration owner (user) must be letters, digits, - or _")
    return ROOT / "configs" / "hands" / "calibration" / f"{user}_{check_side(side)}.yaml"


def hand(station: Station, mode: str, side: str, user: str, log_dir: Path) -> Launch:
    side = check_side(side)
    if side not in station.hands:
        raise UnitError(f"{station.name} has no {side} hand")
    real = _real(mode)
    cal = calibration_file(user, side, fake=not real)
    udp = f"127.0.0.1:{station.sidecars['hand_' + side]},127.0.0.1:{CONSOLE_HAND_PORTS[side]}"
    if real:
        from motion_acq.hand.nova2 import load_gloves

        topic = load_gloves()[side].topic
        command = (f"ros2 run motion_acq_hand hand_node --ros-args -r __node:=motion_acq_hand_{side} "
                   f"-p side:={side} -p glove_topic:={topic} -p calibration:={cal} "
                   f"-p udp_target:={udp} -p log_dir:={log_dir}")
    else:
        command = (f"ros2 launch motion_acq_hand fake_hand.launch.py side:={side} calibration:={cal} "
                   f"udp_target:={udp} enable_on_start:=false")
    side_ko = "오른손" if side == "right" else "왼손"
    return Launch(f"hand_{side}", f"{side_ko} 노드", ros_argv(command), _ros_env(station, mode),
                  summary=f"{side_ko}: 꺼진 채로 뜬다(손은 안 움직인다). 켜기는 따로 승인.", stop_grace_s=15.0)


def hand_driver(station: Station, mode: str, side: str) -> Launch:
    """RH56F1 EtherCAT driver of one hand, exactly as s2r runs it (rh56f1_driver.py --side <side>).

    It goes to OP and holds the fingers where they are until the first command; SIGINT
    takes the master back to INIT (s2r rh56f1_hand_down.sh does the same)."""
    side = check_side(side)
    if not _real(mode):
        raise UnitError("fake: 손 노드가 가짜 드라이버를 같이 띄운다")
    script = station.hand_driver
    if script is None or not script.exists():
        raise UnitError(f"손 드라이버 없음: hands.driver ({script}) 를 확인할 것")
    import os

    rc_ws = os.environ.get("ROBOT_CONTROL_WS", str(Path.home() / "rl_ws/robot_control/ros_ws/install"))
    distro = os.environ.get("ROS_DISTRO") or next(
        (d for d in ("humble", "jazzy") if Path(f"/opt/ros/{d}/setup.bash").exists()), "humble")
    command = (f"set +u; source /opt/ros/{distro}/setup.bash; source {rc_ws}/setup.bash; "
               f"exec python3 {script} --side {side}")
    side_ko = "오른손" if side == "right" else "왼손"
    return Launch(f"ecat_{side}", f"로봇 {side_ko} 드라이버(EtherCAT)", ("bash", "-c", command), _ros_env(station, mode),
                  moves_robot=True, summary=f"{side_ko} EtherCAT 드라이버: OP 로 올라가고, 첫 명령 전에는 손가락이 제자리.",
                  stop_grace_s=10.0)


def hand_enable(station: Station, mode: str, side: str, on: bool) -> Launch:
    side = check_side(side)
    value = "true" if on else "false"
    command = f"ros2 topic pub --once /motion_acq/hand_{side}/enable std_msgs/msg/Bool '{{data: {value}}}'"
    side_ko = "오른손" if side == "right" else "왼손"
    return Launch(f"task_hand_{'on' if on else 'off'}_{side}", f"{side_ko} {'켜기' if on else '끄기'}",
                  ros_argv(command), _ros_env(station, mode), moves_robot=_real(mode) and on,
                  summary=(f"{side_ko}: home(펼침)으로 간 뒤 장갑을 따라간다." if on
                           else f"{side_ko}: home(펼침)으로 돌아간 뒤 멈춘다."),
                  stop_grace_s=5.0)


def calibrate(station: Station, mode: str, side: str, user: str, *, rezero: bool = False) -> Launch:
    """Full calibration (every example pose) or the 2 s open-hand re-zero of a saved one."""
    side = check_side(side)
    real = _real(mode)
    out = calibration_file(user, side, fake=not real)  # validates the owner name
    command = f"ros2 run motion_acq_hand calibrate --side {side} --user {user if real else 'fake'} --out {out}"
    if rezero:
        command += " --rezero"
    if not real:
        command += " --fake --yes"
    side_ko = "오른손" if side == "right" else "왼손"
    if rezero:
        return Launch(f"rezero_{side}", f"{side_ko} 편 손 맞춤", ros_argv(command), _ros_env(station, mode), pty=True,
                      summary="손을 펴고 Enter: 저장된 보정을 오늘 장갑에 맞춘다 (2초).", stop_grace_s=3.0)
    return Launch(f"calib_{side}", f"{side_ko} 장갑 보정", ros_argv(command), _ros_env(station, mode), pty=True,
                  summary="예시 자세마다 Enter 로 기록한다. 사용자마다 한 번, 다음부터는 저장된 보정을 쓴다.",
                  stop_grace_s=3.0)


# -- short tasks ---------------------------------------------------------------------------------------

QUEST_TASKS = {"unworn": "헤드셋 벗고 쓰기 (근접 센서 끔)", "worn": "근접 센서 원래대로", "status": "Quest 상태", "app": "HandUMI 앱 모드", "view": "헤드셋 영상 열기 + VR",
               "page": "헤드셋에 영상 페이지 열기", "vr": "VR 시작", "launch": "HandUMI 앱 다시 시작"}
GLOVE_TASKS = {"up": "SenseCom 시작 + 장갑 연결", "driver": "장갑 드라이버 시작", "stop": "장갑 드라이버 정지",
               "status": "장갑 상태"}


def quest_task(station: Station, name: str) -> Launch:
    if name not in QUEST_TASKS:
        raise UnitError(f"unknown Quest task {name!r}")
    return Launch(f"task_quest_{name}", QUEST_TASKS[name], ("bash", str(ROOT / "scripts" / "quest_usb.sh"), name),
                  _base_env(station), stop_grace_s=3.0)


def glove_task(station: Station, mode: str, name: str) -> Launch:
    if name not in GLOVE_TASKS:
        raise UnitError(f"unknown glove task {name!r}")
    if not station.hands:
        raise UnitError(f"{station.name} has no hands")
    argv = ("bash", str(ROOT / "scripts" / "nova2.sh"), name) + (("both",) if name == "up" else ())
    env = _ros_env(station, "real") if name in ("driver", "status") else _base_env(station)
    if not _real(mode) and name not in ("status", "stop"):
        raise UnitError("glove tasks touch the real gloves: switch the console to real mode")
    return Launch(f"task_glove_{name}", GLOVE_TASKS[name], argv, env, stop_grace_s=3.0)


def station_check(station: Station, user: str) -> Launch:
    argv = [MACQ, "station", "--real", "--preflight-only"]
    if station.hands:
        argv += ["--user", user or "op1"]
    env = _base_env(station)
    if station.ros_domain:
        env["ROS_DOMAIN_ID"] = station.ros_domain
    return Launch("task_check", "실기 점검(읽기만)", tuple(argv), env, stop_grace_s=3.0)
