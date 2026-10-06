"""macq console: unit commands, supervisor, phases, probes parsers, approval and the HTTP guard."""

from __future__ import annotations

import asyncio
import socket
import sys
import time
from pathlib import Path

import pytest

from motion_acq.console import phases, probes, units
from motion_acq.console.core import Console
from motion_acq.console.supervisor import Supervisor, left_behind
from motion_acq.sidecar import parse_udp_targets

ARM4090 = units.Station.load("arm4090")
ARM5080 = units.Station.load("arm5080")


# -- units ---------------------------------------------------------------------------

def test_station_facts():
    assert ARM4090.robot == "openarm_rh56f1"
    assert ARM4090.hands == ("right", "left")
    assert ARM4090.ros_domain == "126"
    assert ARM4090.can_ports == {"right": "can0", "left": "can1"}
    assert not ARM4090.can_auto_repair and ARM4090.has_head
    assert ARM5080.hands == () and not ARM5080.has_head and ARM5080.ros_domain is None


def test_unknown_station_is_refused():
    with pytest.raises(units.UnitError):
        units.Station.load("arm9999")


def test_arm_real_command_matches_the_first_real_runs():
    launch = units.arm(ARM4090, "real", "right", 0.5)
    argv = list(launch.argv)
    assert argv[1:3] == ["teleop-real", "--device"]
    assert argv[argv.index("--side") + 1] == "right"
    assert argv[argv.index("--translation-scale") + 1] == "0.5"
    assert "--skip-can-repair" in argv and "--space-start" in argv and "--fake-robot" not in argv
    assert argv[argv.index("--controller-tcp-calibration") + 1].endswith("identity.yaml")
    assert launch.pty and launch.moves_robot and launch.stop_grace_s >= 60


def test_arm_fake_and_arm5080():
    fake = units.arm(ARM4090, "fake", "both", 0.5)
    assert "--fake-robot" in fake.argv and not fake.moves_robot
    real5080 = units.arm(ARM5080, "real", "left", 0.5)
    assert "--skip-can-repair" not in real5080.argv  # arm5080 repairs its own CAN
    assert "openarmv1" in real5080.argv


@pytest.mark.parametrize("side,scale", [("up", 0.5), ("right", 0.0), ("right", 2.0)])
def test_arm_rejects_bad_options(side, scale):
    with pytest.raises(units.UnitError):
        units.arm(ARM4090, "real", side, scale)


def test_head_sends_to_quest_view_console_and_recorder():
    launch = units.head(ARM4090, "real")
    targets = [launch.argv[i + 1] for i, a in enumerate(launch.argv) if a == "--udp-target"]
    assert targets == ["127.0.0.1:47121", "127.0.0.1:47131", "127.0.0.1:47101"]
    assert "--unlock" in launch.argv and launch.moves_robot
    with pytest.raises(units.UnitError):
        units.head(ARM5080, "real")


def test_hand_joins_the_s2r_domain_and_starts_disabled(tmp_path):
    launch = units.hand(ARM4090, "real", "right", "op1", tmp_path)
    assert launch.env["ROS_DOMAIN_ID"] == "126"
    assert "127.0.0.1:47111,127.0.0.1:47141" in launch.argv[-1]
    assert "enable_on_start" not in launch.argv[-1] and not launch.moves_robot
    fake = units.hand(ARM4090, "fake", "left", "", tmp_path)
    assert fake.env["ROS_DOMAIN_ID"] == "177" and fake.env["ROS_LOCALHOST_ONLY"] == "1"
    assert "enable_on_start:=false" in fake.argv[-1] and "fake_calibration_left.yaml" in fake.argv[-1]
    with pytest.raises(units.UnitError):
        units.hand(ARM5080, "real", "right", "op1", tmp_path)


def test_hand_enable_only_on_needs_approval():
    assert units.hand_enable(ARM4090, "real", "right", True).moves_robot
    assert not units.hand_enable(ARM4090, "real", "right", False).moves_robot


@pytest.mark.parametrize("user", ["", "a b", "x;rm", "../x"])
def test_calibration_owner_is_validated(user):
    with pytest.raises(units.UnitError):
        units.calibration_file(user, "right")


def test_record_requires_only_running_streams(tmp_path):
    assert units.record_streams(ARM4090, {"head", "arm"}) == {"head": 47101}
    assert units.record_streams(ARM4090, set()) == {}
    launch = units.record(ARM4090, "real", "right", 0.5, streams={}, task="pour", episodes=2, output_dir=tmp_path)
    assert "--no-sidecars" in launch.argv and launch.moves_robot


def test_glove_tasks_need_real_mode():
    with pytest.raises(units.UnitError):
        units.glove_task(ARM4090, "fake", "up")
    assert units.glove_task(ARM4090, "real", "driver").env["ROS_DOMAIN_ID"] == "126"


def test_parse_udp_targets():
    assert parse_udp_targets("127.0.0.1:47111,127.0.0.1:47141") == [("127.0.0.1", 47111), ("127.0.0.1", 47141)]
    assert parse_udp_targets("") == []
    with pytest.raises(ValueError):
        parse_udp_targets("127.0.0.1")


# -- supervisor -------------------------------------------------------------------------

CHILD = """
import signal, sys, time, tty
tty.setcbreak(sys.stdin.fileno())  # like KeyboardSpaceListener: single keys, no Enter
print("ready", flush=True)
signal.signal(signal.SIGINT, lambda *a: (print("returning home", flush=True), time.sleep(0.3), sys.exit(0)))
c = sys.stdin.read(1)
print("got", repr(c), flush=True)
time.sleep(30)
"""


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_supervisor_pty_keys_and_single_sigint(tmp_path):
    sup = Supervisor(tmp_path, tmp_path / "running.json")
    child = sup.start("arm", [sys.executable, "-c", CHILD], cwd=tmp_path, env={}, pty=True)
    assert _wait(lambda: "ready" in sup.log_tail("arm"))
    assert left_behind(tmp_path / "running.json")[0]["pid"] == child.pid
    sup.send_key("arm", " ")
    assert _wait(lambda: any("got ' '" in line for line in sup.log_tail("arm")))
    assert sup.stop("arm") is True
    assert sup.stop("arm") is False  # a second SIGINT would cut the safe-pose return short
    assert _wait(lambda: not child.alive())
    assert "returning home" in sup.log_tail("arm")
    assert sup.snapshot("arm")["rc"] == 0
    assert _wait(lambda: left_behind(tmp_path / "running.json") == [])
    with pytest.raises(RuntimeError):
        sup.send_key("arm", " ")


def test_supervisor_refuses_a_second_copy(tmp_path):
    sup = Supervisor(tmp_path)
    sup.start("x", [sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, env={}, pty=False)
    with pytest.raises(RuntimeError):
        sup.start("x", [sys.executable, "-c", "pass"], cwd=tmp_path, env={}, pty=False)
    sup.force_stop("x", grace_s=2.0)
    assert not sup.is_running("x")


# -- phases and parsers ----------------------------------------------------------------------

def test_arm_phase_follows_the_latest_line():
    lines = ["Starting tracking before moving real arms.", "OpenArm right: following the stored path (rest -> home), 8.0 s.",
             "Real openarm_rh56f1 is at home. Open a HandUMI gripper ..."]
    assert phases.phase("arm", lines) == ("ok", "home 대기: Space 로 따라가기 시작")
    assert phases.phase("arm", lines + ["Stopping."])[1].startswith("정지")
    assert phases.phase("head", lines) is None


def test_prompt_detection():
    line = "*** OpenArm right 40 deg from rest. Support the arm, then press Enter to switch the motors off. ***"
    assert phases.prompt([line], "") == line
    assert phases.prompt([], "Override it? [Y/n] ") == "Override it? [Y/n] "
    assert phases.prompt(["Teleop timer started."], "") is None


def test_probe_parsers():
    assert probes.parse_adb_devices("List of devices attached\n2G0YC1ZF8B0B2W\tdevice\n\n") == ["device"]
    assert probes.parse_adb_devices("List of devices attached\n") == []
    link = probes.parse_ip_link("4: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 72 state UP\n  bitrate 1000000 dbitrate 5000000")
    assert link == {"exists": True, "up": True, "fd": True}
    assert probes.parse_topic_info("Type: x\nPublisher count: 1\nSubscription count: 2\n") == (1, 2)
    assert probes.parse_sections("@@list\n/a\n/b\n@@right\nPublisher count: 0") == {
        "list": "/a\n/b", "right": "Publisher count: 0"}
    assert probes.bt_connected("Device X\n\tConnected: yes\n") and not probes.bt_connected("\tConnected: no")
    ss = 'LISTEN 0 128 127.0.0.1:65432 0.0.0.0:* users:(("adb",pid=4321,fd=7))'
    assert probes.parse_ss_listener(ss) == {"name": "adb", "pid": 4321}
    assert probes.parse_ss_listener("") is None


def test_external_quest_source_blocks_the_real_arm(console, monkeypatch):
    from motion_acq.console import gates

    console.set_mode("real")
    data = _fresh({"quest_port": {"name": "macq", "pid": 99}, "quest": {"forward": False},
                   "can": {"right": {"up": True, "fd": True}}})
    monkeypatch.setattr(console, "probe_raw", lambda name: data.get(name))
    assert gates.quest_source(console) == "external"
    assert any("PID 99" in b for b in console.blockers("arm"))


def test_publisher_nodes_from_topic_info():
    text = """Type: rh56f1_interfaces/msg/SetAngle1

Publisher count: 1

Node name: motion_acq_hand_right
Node namespace: /
Topic type: rh56f1_interfaces/msg/SetAngle1
Endpoint type: PUBLISHER
GID: 01.0f

Subscription count: 1

Node name: rh56f1_ecat_right
Node namespace: /
Endpoint type: SUBSCRIPTION
"""
    assert probes.parse_publisher_nodes(text) == ["motion_acq_hand_right"]
    assert probes.parse_topic_info(text) == (1, 1)


def test_hand_command_publishers_cover_angle_target():
    """10.06: the hand node sends angle_target; [켜기] waited for it on angle_set and timed out."""
    target = "Publisher count: 1\n\nNode name: motion_acq_hand_left\nEndpoint type: PUBLISHER\n\nSubscription count: 1\n"
    pos = "Publisher count: 1\n\nNode name: pd_node_left\nEndpoint type: PUBLISHER\n\nSubscription count: 1\n"
    assert probes.command_publishers({"left_angle_target": target}, "left") == (1, ["motion_acq_hand_left"], 1)
    assert probes.command_publishers({"left_angle_target": target, "left_angle_set": pos}, "left") == (
        2, ["pd_node_left", "motion_acq_hand_left"], 1)


# -- console model -------------------------------------------------------------------------------

@pytest.fixture
def console(tmp_path):
    return Console(ARM4090, "fake", log_root=tmp_path, probes=False, listen=False)


def test_arm_needs_a_quest_source(console):
    result = console.start("arm")
    assert not result["ok"] and "Quest" in result["error"]


def test_real_motion_needs_the_operator_confirmation(console, monkeypatch):
    console.set_mode("real")
    monkeypatch.setattr(console, "blockers", lambda key: [])
    started = []
    monkeypatch.setattr(console, "_spawn", lambda launch, confirmed=False: started.append(launch) or {"ok": True})
    reply = console.start("arm", {"arm_side": "left"})
    assert reply["need_confirm"] and "--side left" in reply["command"] and reply["token"]
    assert started == [] and console.settings["arm_side"] == "left"
    wrong = console.start("arm", confirm=True, token="0" * 16)
    assert wrong["need_confirm"] and started == []  # a guessed token never starts
    changed = console.start("arm", {"arm_side": "right"}, confirm=True, token=wrong["token"])
    assert changed["need_confirm"] and "--side right" in changed["command"] and started == []
    ok = console.start("arm", confirm=True, token=changed["token"])
    assert ok == {"ok": True} and "--side" in started[0].argv and started[0].argv[started[0].argv.index("--side") + 1] == "right"
    again = console.start("arm", confirm=True, token=changed["token"])
    assert again["need_confirm"]  # a token is used once


def test_mode_switch_drops_pending_confirmations(console, monkeypatch):
    console.set_mode("real")
    monkeypatch.setattr(console, "blockers", lambda key: [])
    started = []
    monkeypatch.setattr(console, "_spawn", lambda launch, confirmed=False: started.append(launch) or {"ok": True})
    token = console.start("head")["token"]
    console.set_mode("fake")
    console.set_mode("real")
    assert console.start("head", confirm=True, token=token)["need_confirm"] and started == []


def test_record_output_dir_may_differ_between_ask_and_confirm():
    from motion_acq.console.core import same_command

    a = units.record(ARM4090, "real", "right", 0.5, streams={}, task="t", episodes=1, output_dir=Path("/o/1"))
    b = units.record(ARM4090, "real", "right", 0.5, streams={}, task="t", episodes=1, output_dir=Path("/o/2"))
    c = units.record(ARM4090, "real", "left", 0.5, streams={}, task="t", episodes=1, output_dir=Path("/o/1"))
    assert same_command(a, b) and not same_command(a, c)


def _fresh(data: dict) -> dict:
    return {"fast_at": time.time(), "quest": {"forward": True}, "can_holders": [], **data}


def test_real_arm_blockers(console, monkeypatch):
    console.set_mode("real")
    probes_now = _fresh({"can": {"right": {"port": "can0", "up": True, "fd": True},
                                 "left": {"port": "can1", "up": False, "fd": False}}})
    monkeypatch.setattr(console, "probe_raw", lambda name: probes_now.get(name))
    console.update_settings({"arm_side": "right"})
    assert any("오른손 노드" in b for b in console.blockers("arm"))  # 10.06: the arm closes the hand first
    monkeypatch.setattr(console.sup, "is_running", lambda key: key in ("hand_right", "hand_left"))
    assert console.blockers("arm") == []
    console.update_settings({"arm_side": "both"})
    assert any("can1" in b for b in console.blockers("arm"))
    probes_now["can_holders"] = ["1234 ros2_control_node --ros-args"]
    assert any("s2r" in b for b in console.blockers("arm"))
    probes_now["can_holders"] = {"error": "boom"}  # a failed probe never passes as free
    assert any("확인 실패" in b for b in console.blockers("arm"))
    probes_now["fast_at"] = time.time() - 60
    assert any("오래됨" in b for b in console.blockers("arm"))


def test_arm_and_record_exclude_each_other(console, monkeypatch):
    monkeypatch.setattr(console.sup, "is_running", lambda key: key in ("record", "mock_quest"))
    assert any("record" in b for b in console.blockers("arm"))


def test_left_behind_units_block_their_slot(console, monkeypatch):
    monkeypatch.setattr(console, "left_keys", lambda: {"record"})
    monkeypatch.setattr(console.sup, "is_running", lambda key: key == "mock_quest")
    assert any("record" in b for b in console.blockers("arm"))
    assert any("이전 콘솔" in b for b in console.blockers("record"))


def test_hand_blockers_follow_the_s2r_driver(console, monkeypatch):
    from motion_acq.console import gates

    console.set_mode("real")
    hand = {"driver": True, "angle_set_publishers": 0, "angle_set_publisher_nodes": []}
    ros = {"domain": "126", "at": time.time(), "glove_topics": {"right": True}, "hands": {"right": hand}}
    assert gates.hand_blockers(console, "right", ros, starting=True) == []
    hand.update(angle_set_publishers=1, angle_set_publisher_nodes=["pd_node_right"])  # s2r commands the hand
    assert any("pd_node_right" in b for b in gates.hand_blockers(console, "right", ros, starting=True))
    hand.update(angle_set_publishers=1, angle_set_publisher_nodes=["motion_acq_hand_right"])
    assert gates.hand_blockers(console, "right", ros, starting=False) == []
    hand.update(angle_set_publishers=2, angle_set_publisher_nodes=["motion_acq_hand_right", "pd_node_right"])
    assert gates.hand_blockers(console, "right", ros, starting=False)
    hand.update(angle_set_publishers=1, angle_set_publisher_nodes=[])  # names not read: never assume ours
    assert gates.hand_blockers(console, "right", ros, starting=False)
    ros["domain"] = "177"  # left over from fake mode
    assert any("확인 중" in b for b in gates.hand_blockers(console, "right", ros, starting=True))
    ros.update(domain="126", hands={"right": {"driver": False}})
    assert any("s2r 콘솔" in b for b in gates.hand_blockers(console, "right", ros, starting=True))


def test_closing_console_starts_nothing(console):
    console.begin_shutdown()
    assert "종료" in console.start("mock_quest")["error"]


def test_mode_change_refused_while_running(console, monkeypatch):
    monkeypatch.setattr(console.sup, "running", lambda: ["head"])
    with pytest.raises(units.UnitError):
        console.set_mode("real")


def test_snapshot_shape(console):
    snap = console.snapshot()
    assert snap["mode"] == "fake" and snap["station"]["name"] == "arm4090"
    assert {"arm", "record", "head", "hand_right", "calib_left", "quest_view", "mock_quest"} <= set(snap["units"])


# -- HTTP guard -------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_guard(console):
    from aiohttp import ClientSession
    from aiohttp.test_utils import TestServer

    from motion_acq.console.server import build_app

    async def body():
        port = _free_port()
        server = TestServer(build_app(console, port=port), host="127.0.0.1", port=port)
        await server.start_server()
        base = f"http://127.0.0.1:{port}"
        try:
            async with ClientSession() as http:
                async with http.get(f"{base}/api/ping") as r:
                    assert r.status == 200 and (await r.json())["app"] == "macq-console"
                async with http.get(f"{base}/api/ping", headers={"Host": "evil.example"}) as r:
                    assert r.status == 403
                async with http.post(f"{base}/api/stop_all", json={}) as r:
                    assert r.status == 403  # no X-Macq-Console header
                hdr = {"X-Macq-Console": "1"}
                async with http.post(f"{base}/api/stop_all", json={}, headers={**hdr, "Origin": "http://evil"}) as r:
                    assert r.status == 403
                async with http.post(f"{base}/api/stop_all", json={}, headers=hdr) as r:
                    assert r.status == 200 and (await r.json())["ok"]
                async with http.post(f"{base}/api/start", json={"key": "nope"}, headers=hdr) as r:
                    assert not (await r.json())["ok"]
                async with http.get(f"{base}/") as r:
                    assert "macq" in await r.text()
        finally:
            await server.close()

    asyncio.run(body())
    assert (Path(console.log_root) / "intents.jsonl").exists()


# -- simplified flow (body view) -----------------------------------------------------------------------

def test_quest_view_fake_is_a_test_pattern_without_the_pose_server():
    assert units.quest_view(ARM4090, "fake").argv[-2:] == ("--test-pattern", "--no-pose-server")
    assert units.quest_view(ARM4090, "real").argv[-1] == "quest-view"


def test_record_takes_the_head_camera_from_quest_view(tmp_path):
    launch = units.record(ARM4090, "real", "both", 0.5, streams={"head": 47101}, task="t", episodes=1,
                          output_dir=tmp_path, cameras=("head",))
    argv = list(launch.argv)
    assert "--skip-cameras" not in argv and argv[argv.index("--cameras") + 1] == "head"
    assert ARM4090.cameras == ("head",) and ARM5080.cameras == ()


def test_quest_source_follows_the_mode(console, monkeypatch):
    from motion_acq.console import gates

    monkeypatch.setattr(console.sup, "is_running", lambda key: key == "quest_view")
    assert gates.quest_source(console) is None  # fake quest-view = test pattern, no poses
    console.set_mode("real")
    assert gates.quest_source(console) == "view"


def test_webxr_arm_needs_the_direction_check(console, monkeypatch):
    console.set_mode("real")
    data = _fresh({"can": {"right": {"port": "can0", "up": True, "fd": True}}})
    monkeypatch.setattr(console, "probe_raw", lambda name: data.get(name))
    monkeypatch.setattr(console.sup, "is_running", lambda key: key in ("quest_view", "hand_right", "hand_left"))
    assert any("방향 확인" in b for b in console.blockers("arm"))
    monkeypatch.setattr(console.direction, "stop", lambda: None)
    assert console.action("direction:ok")["ok"] and console.settings["webxr_arm_ok"]
    assert console.blockers("arm") == []
    console.action("direction:reset")
    assert any("방향 확인" in b for b in console.blockers("arm"))


def test_hand_on_asks_once_for_driver_node_and_enable(console, monkeypatch):
    console.set_mode("real")
    monkeypatch.setattr(console, "_hand_precheck", lambda side: [])
    monkeypatch.setattr(console, "_hand_driver_present", lambda side: False)
    console.update_settings({"user": "op1"})
    jobs = []
    monkeypatch.setattr(console, "_job", lambda name, fn: jobs.append(name) or {"ok": True})
    reply = console.action("hand_on:right")
    assert reply["need_confirm"] and jobs == []
    lines = reply["command"].split("\n")
    assert "rh56f1_driver.py --side right" in lines[0] and "hand_node" in lines[1] and "enable" in lines[2]
    assert reply["summary"].startswith("드라이버 → 노드 → 켜기")
    assert console.action("hand_on:right", confirm=True, token=reply["token"]) == {"ok": True}
    assert jobs == ["hand_right"]


def test_hand_on_uses_a_running_driver(console, monkeypatch):
    console.set_mode("real")
    monkeypatch.setattr(console, "_hand_precheck", lambda side: [])
    monkeypatch.setattr(console, "_hand_driver_present", lambda side: True)
    console.update_settings({"user": "op1"})
    reply = console.action("hand_on:left")
    assert "rh56f1_driver.py" not in reply["command"] and reply["summary"].startswith("노드 → 켜기")


def test_hand_driver_command_matches_s2r(tmp_path):
    launch = units.hand_driver(ARM4090, "real", "left")
    assert launch.key == "ecat_left" and launch.moves_robot and launch.env["ROS_DOMAIN_ID"] == "126"
    assert launch.argv[-1].endswith("rh56f1_driver.py --side left")
    with pytest.raises(units.UnitError):
        units.hand_driver(ARM4090, "fake", "left")


def test_driver_off_refused_while_the_hand_node_runs(console, monkeypatch):
    console.set_mode("real")
    monkeypatch.setattr(console.sup, "is_running", lambda key: key in ("hand_right", "ecat_right"))
    assert "끄기" in console.action("driver_off:right")["error"]


def test_stop_all_stops_hands_only_after_the_arms(console, monkeypatch):
    import threading as _threading

    state = {"arm": True, "hand_right": True, "ecat_right": True, "head": True}
    signals = []

    def stop(key):
        signals.append(key)
        if key.startswith("hand_"):
            state[key] = False  # the node opens the hand and exits
        return True

    monkeypatch.setattr(console.sup, "running", lambda: [k for k, v in state.items() if v])
    monkeypatch.setattr(console.sup, "is_running", lambda key: state.get(key, False))
    monkeypatch.setattr(console.sup, "stop", stop)
    started = []
    monkeypatch.setattr(_threading, "Thread", lambda target, **kw: started.append(target) or
                        type("T", (), {"start": lambda self: None})())
    reply = console.stop_all()
    assert set(reply["stopped"]) == {"arm", "head"} and set(reply["later"]) == {"hand_right", "ecat_right"}
    assert "hand_right" not in signals
    state.update(arm=False, head=False)  # the arm reached rest
    started[0]()
    assert signals[-2:] == ["hand_right", "ecat_right"]


def test_record_checklist_lists_every_stream(console, monkeypatch):
    monkeypatch.setattr(console.sup, "running", lambda: ["mock_quest", "quest_view", "head"])
    monkeypatch.setattr(console.sup, "is_running", lambda key: key in ("mock_quest", "quest_view", "head"))
    items = {i["name"]: i["ok"] for i in console.record_checklist()}
    assert items["Quest 자세"] and items["카메라 영상"] and items["목"] and items["로봇 팔"]
    assert not items["로봇 오른손"] and not items["로봇 왼손"]


def test_direction_describe():
    import numpy as np

    from motion_acq.console.direction import describe

    assert describe(np.array([0.12, 0.01, 0.0]))["main"] == "앞 12 cm"
    assert describe(np.array([0.0, -0.05, 0.01]))["main"] == "오른 5 cm"
    assert describe(np.array([0.0, 0.0, 0.005]))["main"] == "거의 그대로"


STREAMER_FAILURE_10_04 = """[21:24:17] ERROR - OpenArm command streamer failed: OpenArm right joint7 following error 0.351 rad exceeds 0.350 rad.
[21:24:17] ERROR - Arms not returned home (streamer failed: OpenArmJointStreamer failed).
Traceback (most recent call last):
  File "/home/user/rl_ws/motion_acq/src/motion_acq/real/openarm/driver.py", line 467, in _run
    raise RuntimeError(
RuntimeError: OpenArm right joint7 following error 0.351 rad exceeds 0.350 rad.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/user/rl_ws/motion_acq/src/motion_acq/real/streamer.py", line 179, in raise_if_failed
    raise RuntimeError(f"{type(self).__name__} failed") from self._error
RuntimeError: OpenArmJointStreamer failed""".splitlines()


def test_error_line_is_the_root_cause_of_a_chained_traceback():
    """10.04: the console showed 'OpenArmJointStreamer failed' instead of the following error."""
    assert phases.error_line(STREAMER_FAILURE_10_04) == "OpenArm right joint7 following error 0.351 rad exceeds 0.350 rad."
    single = ["Traceback (most recent call last):", '  File "x.py", line 1, in <module>',
              "motion_acq.real.openarm.home_path.HomePathError: right j5 15.5 deg from rest"]
    assert phases.error_line(single) == "right j5 15.5 deg from rest"
    assert phases.error_line(["SystemExit: no glove samples"]) == "no glove samples"


def test_untracked_headset_at_space_is_shown():
    lines = ["Space pressed; starting right.",
             "[22:30:01] WARNING - Headset not tracked: right not started (robot forward comes from the headset heading)."]
    assert phases.phase("arm", lines) == ("warn", "헤드셋 추적 안 됨: 팔 시작 안 함 (헤드셋을 쓰거나, 벗고 쓰면 깨운 채 로봇 앞을 보게 두고 Space 다시)")


def test_error_line_ignores_an_earlier_caught_traceback():
    caught = ["[21:22:56] ERROR - Rerun logging failed", "Traceback (most recent call last):",
              '  File "record.py", line 964, in log', "ConnectionError: rerun viewer closed",
              "[21:23:00] INFO - Teleop timer started."]
    assert phases.error_line(caught + STREAMER_FAILURE_10_04) == (
        "OpenArm right joint7 following error 0.351 rad exceeds 0.350 rad.")


def test_error_line_finds_the_cause_when_the_tail_starts_inside_it():
    cut = STREAMER_FAILURE_10_04[STREAMER_FAILURE_10_04.index("    raise RuntimeError("):]
    assert phases.error_line(cut) == "OpenArm right joint7 following error 0.351 rad exceeds 0.350 rad."


def test_hand_link_closes_the_hands_before_the_arm_path(monkeypatch):
    """10.06 user: fist -> arm home -> hand open. A real UDP round trip with a scripted hand node."""
    import json
    import socket
    import threading

    from motion_acq.hand_link import HandLink, HandLinkError, hand_link_reply, parse_arm_message

    hand = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hand.bind(("127.0.0.1", 0))
    hand.settimeout(0.05)
    port = hand.getsockname()[1]
    seen, closed_after = [], 3   # the hand reports the fist on its 3rd message
    stop = threading.Event()

    def node():
        while not stop.is_set():
            try:
                data, sender = hand.recvfrom(4096)
            except OSError:
                continue
            seen.append(parse_arm_message(data))
            hand.sendto(hand_link_reply("right", at_rest=len(seen) >= closed_after, mode="enabled",
                                        phase="rest", fault=None), sender)

    thread = threading.Thread(target=node, daemon=True)
    thread.start()
    link = HandLink({"right": port}, wait_s=2.0, answer_s=1.0, beat_s=0.05)
    try:
        link.require_rest(["right", "left"])        # left has no link: ignored
        assert seen and set(seen) == {"path"}
        link.announce("home", ["right"])
        time.sleep(0.2)
        assert seen[-1] == "home"
    finally:
        link.close()
        stop.set()
        thread.join(1.0)
        hand.close()
    silent = HandLink({"right": port}, wait_s=1.0, answer_s=0.2, beat_s=0.05)
    try:
        with pytest.raises(HandLinkError, match="응답하지"):
            silent.require_rest(["right"])
    finally:
        silent.close()
    assert parse_arm_message(json.dumps({"arm": "dance"}).encode()) is None
