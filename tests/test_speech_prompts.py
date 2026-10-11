"""Calibration read aloud (10.11 user: "보정 절차를 누르면 소리로 알려주면 좋겠는데").

The operator wears the Nova 2 gloves and the Quest during calibration, so the
console prompts and the SDK glove calibration steps are spoken in Korean.
"""

from __future__ import annotations

import pytest

from motion_acq.hand.direct import sdk_calibration_speech
from motion_acq.utils import speech
from motion_acq.utils.speech import Voice, speech_text


def test_a_pose_prompt_is_read_without_the_screen_details():
    line = speech_text("[오른손 3/11] 주먹: 네 손가락을 끝까지 말아 쥐고, 엄지도 굽혀 검지·중지 위를 덮는다."
                       " 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
    assert line.startswith("오른손 3번. 주먹:")
    assert "검지, 중지" in line and "엔터를 누르세요" in line and "Enter" not in line


@pytest.mark.parametrize("text, heard", [
    ("  기록 실패 (no samples (0 < 20)): 같은 자세를 다시 합니다 (1/3)", "기록 실패: 같은 자세를 다시 합니다"),
    ("[왼손 손가락별 2/5] 엄지-검지 집기 (x)", "왼손 손가락별 2번. 엄지 검지 집기"),
    ("  기록됨 (120 샘플)", "기록됨"),
])
def test_numbers_in_parentheses_are_not_read(text, heard):
    assert speech_text(text) == heard


@pytest.fixture()
def spoken(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech.subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr(speech.subprocess, "Popen", lambda cmd, **kw: calls.append(list(cmd)))
    return calls


def test_a_start_cue_cuts_the_prompt_still_being_read(spoken):
    voice = Voice(True, lang="ko", now_cues=("기록 중", "시작:"))
    voice("[오른손 1/11] 손 펴기: 다섯 손가락을 쭉 편다. 자세를 잡고 멈춘 뒤 Enter 를 누르세요 ")
    voice("  기록 중: 그대로 멈춰 있으세요")
    assert spoken[0][:3] == ["spd-say", "-l", "ko"]
    assert spoken[1] == ["spd-say", "-C"]
    assert spoken[2] == ["spd-say", "-l", "ko", "기록 중: 그대로 멈춰 있으세요"]


def test_a_muted_voice_says_nothing(spoken):
    Voice(False, lang="ko")("  기록 중: 그대로 멈춰 있으세요")
    assert spoken == []


def test_the_sdk_calibration_steps_are_read_once_each():
    move = "Move Fingers | Calibration State: Move Fingers"
    done = "All Sensors Moved | Calibration State: All Sensors Moved"
    assert sdk_calibration_speech("left", None, move).startswith("왼손 장갑 보정.")
    assert sdk_calibration_speech("left", move, move.replace("State:", "State :")) is None
    assert sdk_calibration_speech("right", move, done) == "오른손 장갑 보정 완료"
    assert sdk_calibration_speech("right", done, "Unknown | Calibration State: All Sensors Moved") is None
