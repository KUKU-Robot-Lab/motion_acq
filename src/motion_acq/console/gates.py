"""What blocks a unit from starting (Korean reasons for the operator; empty list = may start).

Real mode is strict: a probe that failed, has not run yet or is older than
FAST_STALE_S blocks too, so a missing reading never passes as "free". The
hand checks read ROS on the domain the hand nodes use and name the nodes
publishing /hand_<side>/angle_set: only our own motion_acq_hand_<side> may.
"""

from __future__ import annotations

import time

from motion_acq.console import units

# arm and record both own the CAN arms. quest_view and the mock never clash: the mock only runs in
# fake mode, where quest-view serves a test pattern without the pose server.
EXCLUSIVE = ({"arm", "record"},)
QUEST_PRODUCERS = ("head", "arm", "record")
FAST_STALE_S = 10.0
ROS_STALE_S = 20.0


def quest_source(console) -> str | None:
    """Where the HandUMI pose stream on TCP 65432 comes from."""
    if console.sup.is_running("mock_quest"):
        return "mock"
    if console.mode == "real" and console.sup.is_running("quest_view"):
        return "view"
    owner = console.probe("quest_port") or {}
    if owner.get("name") and owner.get("name") != "adb":
        return "external"  # a quest-view / mock not started by this console
    return "app" if console.probe("quest").get("forward") else None


def external_quest(console) -> str:
    owner = console.probe("quest_port") or {}
    return f"TCP 65432 를 이 콘솔 밖의 {owner.get('name', '?')} (PID {owner.get('pid', '?')}) 가 쓰는 중"


def _running_or_left(console, key: str) -> bool:
    return console.sup.is_running(key) or key in console.left_keys()


def blockers(console, key: str) -> list[str]:
    out = []
    if console.closing:
        return ["콘솔이 종료 중이다"]
    if key in console.left_keys():
        out.append(f"이전 콘솔이 남긴 {key} 가 아직 돈다: 위 빨간 막대에서 먼저 정지")
    for group in EXCLUSIVE:
        if key in group:
            out += [f"{other} 실행 중" for other in group - {key} if _running_or_left(console, other)]
    source = quest_source(console)
    if key in QUEST_PRODUCERS and source is None:
        out.append("Quest 자세 입력 없음: HandUMI 앱 모드" + (" 또는 가짜 Quest" if console.mode == "fake" else ""))
    if console.mode == "real":
        if key in QUEST_PRODUCERS:
            out += _fast_probe_problems(console)
        if key in ("arm", "record"):
            out += _arm_blockers(console, source)
        if key == "head":
            out += _head_blockers(console)
        if key.startswith("hand_"):
            out += hand_blockers(console, key[5:], console.probe("ros"), starting=True)
        if key.startswith("ecat_"):
            out += driver_blockers(console, key[5:])
        if key.startswith(("calib_", "rezero_")):
            side = key.split("_", 1)[1]
            ros = fresh_ros(console, console.probe("ros"))
            if ros is None or not (ros.get("glove_topics") or {}).get(side):
                out.append("장갑 토픽 없음: 장갑 연결과 드라이버 먼저")
            if key.startswith("rezero_") and not (console.calibration().get(side) or {}).get("ok"):
                out.append("저장된 보정이 없다: 먼저 [보정]")
    elif key.startswith("hand_") and not units.calibration_file("", key[5:], fake=True).exists():
        out.append(f"fake 보정 파일 없음: 먼저 [{key[5:]} 보정] 또는 scripts/fake_hand_check.sh {key[5:]}")
    return out


def _fast_probe_problems(console) -> list[str]:
    at = console.probe_raw("fast_at")
    if not isinstance(at, (int, float)) or time.time() - at > FAST_STALE_S:
        return ["장치 상태를 아직 못 읽었다(또는 오래됨): 잠시 뒤 다시"]
    return []


def _arm_blockers(console, source: str | None) -> list[str]:
    out = []
    if source == "view" and not console.settings.get("webxr_arm_ok"):
        out.append("헤드셋 영상(WebXR)으로 팔을 처음 움직이기 전에 머리(Quest) > 방향 확인을 마칠 것(로봇은 안 움직인다)")
    if source == "external":
        out.append(external_quest(console) + ": 그 프로그램을 끄고 HandUMI 앱 모드로")
    side = str(console.settings["arm_side"])
    can = console.probe("can")
    if "error" in can:
        out.append(f"CAN 상태 확인 실패: {can['error']}")
    for s in (("right", "left") if side == "both" else (side,)):
        link = can.get(s) or {}
        if not (link.get("up") and link.get("fd")):
            out.append(f"CAN {link.get('port', s)} 가 UP/FD 가 아님(s2r 콘솔 CAN 단계 또는 운영자 확인)")
    holders = console.probe_raw("can_holders")
    if not isinstance(holders, list):
        out.append("s2r CAN 점유 확인 실패: 잠시 뒤 다시")
    elif holders:
        out.append(f"s2r 가 CAN 을 잡고 있다: {holders[0][:80]}")
    return out


def _head_blockers(console) -> list[str]:
    port = console.probe("head_port")
    if "exists" not in port:
        return [f"머리 포트 확인 실패: {port.get('error', '아직 못 읽음')}"]
    if not port["exists"]:
        return [f"머리 포트 없음: {port.get('path', '?')}"]
    if port.get("holders"):
        return [f"머리 포트를 PID {port['holders']} 가 사용 중(s2r 머리 노드?)"]
    return []


def driver_running_elsewhere(console, side: str) -> list[int]:
    """PIDs of an RH56F1 driver for this hand that this console did not start (s2r)."""
    if console.sup.is_running(f"ecat_{side}"):
        return []
    return list((console.probe("hand_drivers") or {}).get(side) or [])


def driver_blockers(console, side: str) -> list[str]:
    elsewhere = driver_running_elsewhere(console, side)
    if elsewhere:
        return [f"{side} 손 드라이버가 이미 떠 있다(PID {elsewhere}, s2r 콘솔?): 그대로 쓴다"]
    if console.mode != "real":
        return ["fake 에서는 손 노드가 가짜 드라이버를 같이 띄운다"]
    return []


def expected_domain(console) -> str | None:
    return console.station.ros_domain if console.mode == "real" else units.FAKE_DOMAIN


def fresh_ros(console, ros: dict | None) -> dict | None:
    """The ROS probe if it is recent, error-free and on the domain the hand nodes use now."""
    if not ros or "error" in ros or time.time() - float(ros.get("at", 0)) > ROS_STALE_S:
        return None
    return ros if ros.get("domain") == expected_domain(console) else None


def hand_blockers(console, side: str, ros: dict | None, *, starting: bool) -> list[str]:
    ros = fresh_ros(console, ros)
    if ros is None:
        return [f"ROS 상태(도메인 {expected_domain(console)}) 확인 중: 잠시 뒤 다시"]
    hand = (ros.get("hands") or {}).get(side) or {}
    out = []
    if not hand.get("driver"):
        out.append(f"RH56F1 {side} EtherCAT 드라이버가 도메인 {ros.get('domain')} 에 없다: s2r 콘솔에서 켤 것")
    ours = f"motion_acq_hand_{side}"
    nodes = list(hand.get("angle_set_publisher_nodes") or [])
    count = int(hand.get("angle_set_publishers", 0))
    foreign = nodes if starting else [n for n in nodes if n != ours]
    if foreign or (starting and count):
        who = ", ".join(foreign) or f"{count} 개"
        out.append(f"/hand_{side}/angle_set 을 다른 노드({who})가 발행 중(s2r pd?): 그쪽 손 제어를 먼저 끌 것")
    elif not starting and (count != len(nodes) or ours not in nodes):
        out.append(f"/hand_{side}/angle_set 발행자를 확인하지 못했다(우리 노드 {ours} 가 아직 안 보임): 잠시 뒤 다시")
    if not (ros.get("glove_topics") or {}).get(side) and console.mode == "real":
        out.append("장갑 토픽 없음: 장갑 연결과 드라이버 먼저")
    return out
