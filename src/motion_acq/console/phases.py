"""Read a unit's state from its log lines (the arm programs have no status stream).

Each rule is (substring, tone, Korean text); the rule matching the latest line
wins. Tones: ok (green), live (blue, moving/following), warn, bad, mute.
"""

from __future__ import annotations

import re

Rule = tuple[str, str, str]

ARM_RULES: tuple[Rule, ...] = (
    ("Starting tracking", "live", "Quest 트래킹 연결 중"),
    ("Connecting OpenArm", "live", "OpenArm 연결 중"),
    ("aligning to rest", "live", "차렷 자세로 맞추는 중"),
    ("following the stored path", "live", "저장 경로로 이동 중"),
    ("is at home", "ok", "home 대기: Space 로 따라가기 시작"),
    ("Space pressed; starting", "live", "따라가는 중"),
    ("arm anchored", "live", "따라가는 중"),
    ("Tracking lost", "warn", "트래킹 끊김: 팔 정지"),
    ("Tracking recovered", "live", "트래킹 복구"),
    ("Stopping.", "live", "정지: home → 차렷 이동 중"),
    ("Arms at home", "live", "home 도착: 차렷으로 이동"),
    ("Arms ready to disable", "ok", "차렷 도착: 모터 끔"),
    ("back at rest", "ok", "차렷 도착"),
    ("start failed", "bad", "시작 실패: 차렷으로 복귀"),
    ("about to switch off away from rest", "bad", "차렷이 아닌 곳에서 모터를 끄려 함: 팔을 받칠 것"),
    ("Traceback", "bad", "오류로 멈춤 (로그 확인)"),
    ("neither rest", "bad", "차렷도 home 도 아니라 시작 안 함: 팔을 차렷에 맞출 것(모터는 안 켜졌다)"),
    ("Refusing to start", "bad", "시작 거부 (자세히 보기)"),
)

RECORD_RULES: tuple[Rule, ...] = (
    ("Starting tracking", "live", "Quest 트래킹 연결 중"),
    ("following the stored path", "live", "저장 경로로 이동 중"),
    ("Waiting for fresh controller tracking", "warn", "컨트롤러 트래킹 기다림"),
    ("recording gate open", "ok", "준비: Space 로 에피소드 시작"),
    ("--- Episode", "ok", "에피소드 준비: Space 로 시작"),
    ("Recording episode started", "live", "녹화 중: Space 로 저장"),
    ("Space pressed: saving episode", "live", "저장 중"),
    (" saved, ", "ok", "저장됨: home 으로 복귀"),
    ("returning home", "live", "home 으로 복귀 중"),
    ("R pressed", "warn", "다시: home 으로 복귀"),
    ("Returning enabled arms home", "live", "home 으로 복귀 중"),
    ("Q pressed", "warn", "마치는 중"),
    ("Signal received", "live", "정지: 녹화 버리고 home → 차렷"),
    ("finishing recording session", "live", "마치는 중"),
    ("recording session finished", "ok", "녹화 끝, 팔 home"),
    ("integrity validation passed", "ok", "데이터셋 검증 통과"),
    ("Done. Recorded", "ok", "녹화 끝: home → 차렷"),
    ("Arms ready to disable", "ok", "차렷 도착: 모터 끔"),
    ("Tracking lost", "warn", "트래킹 끊김"),
    ("Traceback", "bad", "오류로 멈춤 (로그 확인)"),
)

QUEST_VIEW_RULES: tuple[Rule, ...] = (
    ("page http://localhost", "ok", "대기: 헤드셋에서 페이지 열기"),
    ("headset page connected", "live", "헤드셋 페이지 연결됨"),
    ("headset page left", "warn", "헤드셋 페이지 끊김"),
    ("no headset poses", "warn", "헤드셋 자세가 안 옴: VR 시작 또는 USB 재연결"),
    ("is taken", "bad", "TCP 65432 를 다른 것이 쓰는 중(HandUMI 앱 forward?)"),
)

CALIB_RULES: tuple[Rule, ...] = (
    ("saved", "ok", "보정 저장됨"),
    ("moved", "warn", "움직임: 그 자세 다시"),
)

RULES = {"arm": ARM_RULES, "record": RECORD_RULES, "quest_view": QUEST_VIEW_RULES,
         "calib_right": CALIB_RULES, "calib_left": CALIB_RULES}

PROMPT = re.compile(r"press enter|\[y/n\]|\[Y/n\]|\[y/N\]", re.IGNORECASE)
POSES_IN = re.compile(r"poses in (\d+) \(last ([^)]*) ago\), tracking clients (\d+), camera frames (\d+)")


def phase(key: str, lines: list[str]) -> tuple[str, str] | None:
    rules = RULES.get(key)
    if not rules:
        return None
    for line in reversed(lines):
        for needle, tone, text in rules:
            if needle in line:
                return tone, text
    return None


def prompt(lines: list[str], partial: str) -> str | None:
    """The question the program waits on (Enter / y-n), if the last output is one."""
    if partial and PROMPT.search(partial):
        return partial
    if lines and PROMPT.search(lines[-1]):
        return lines[-1]
    return None


ERROR_LINE = re.compile(r"(?:[A-Za-z_.]+Error|SystemExit|RuntimeError): (.+)$")


def error_line(lines: list[str]) -> str | None:
    """The last 'SomethingError: message' line of a program that stopped on an error."""
    for line in reversed(lines):
        match = ERROR_LINE.search(line)
        if match:
            return match.group(1)[:400]
    return None


def quest_view_stats(lines: list[str]) -> dict | None:
    for line in reversed(lines):
        match = POSES_IN.search(line)
        if match:
            return {"poses": int(match.group(1)), "last": match.group(2),
                    "clients": int(match.group(3)), "frames": int(match.group(4))}
    return None
