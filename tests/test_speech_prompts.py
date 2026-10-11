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
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
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


def test_over_ssh_the_voice_goes_to_the_desktop_session(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(speech.os.path, "isdir", lambda path: True)
    assert speech._session_env()["XDG_RUNTIME_DIR"] == f"/run/user/{speech.os.getuid()}"
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/elsewhere")
    assert speech._session_env()["XDG_RUNTIME_DIR"] == "/elsewhere"


# ---- neural voice (10.12 user: espeak's Korean is hard to understand) ----------------------------


@pytest.fixture()
def neural(monkeypatch, tmp_path):
    """Neural voice on, synthesis replaced by a tone, playback recorded."""
    import numpy as np

    monkeypatch.delenv("MACQ_TTS", raising=False)
    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(speech, "_offline_until", 0.0)
    made, played = [], []

    def fake_wav(text, voice):
        path = speech.cache_path(text, voice)
        if not path.exists():
            made.append(text)
            import wave
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(speech.SAMPLE_RATE)
                out.writeframes(np.zeros(240, dtype=np.int16).tobytes())
        return path

    class Proc:
        def __init__(self, cmd, **kw):
            played.append(cmd)

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(speech, "neural_wav", fake_wav)
    monkeypatch.setattr(speech.subprocess, "Popen", Proc)
    monkeypatch.setattr(speech.subprocess, "run", lambda cmd, **kw: played.append(list(cmd)))
    return made, played


def test_korean_uses_the_neural_voice_and_reuses_its_cache(neural):
    made, played = neural
    speech.say("정면 저장", lang="ko", blocking=True)
    speech.say("정면 저장", lang="ko", blocking=True)
    assert made == ["정면 저장"]
    paplays = [cmd for cmd in played if cmd[0] == "paplay"]
    assert len(paplays) == 2 and paplays[0][1].endswith(".wav")


def test_without_the_network_korean_falls_back_to_espeak(monkeypatch, neural):
    _, played = neural
    monkeypatch.setattr(speech, "neural_wav", lambda text, voice: None)
    speech.say("정면 저장", lang="ko", blocking=True)
    assert ["spd-say", "-l", "ko", "정면 저장", "--wait"] in played


def test_english_and_forced_espeak_keep_the_old_voice(monkeypatch, neural):
    assert speech.neural_voice(None) is None
    monkeypatch.setenv("MACQ_TTS", "espeak")
    assert speech.neural_voice("ko") is None


def test_prefetch_makes_each_prompt_once(neural):
    made, _ = neural
    assert speech.prefetch(["십", "구", "십"]) == 3
    assert made == ["십", "구"]


def test_a_failed_synthesis_waits_before_trying_the_network_again(monkeypatch, tmp_path):
    import sys
    import types

    monkeypatch.delenv("MACQ_TTS", raising=False)
    monkeypatch.setattr(speech, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(speech, "_offline_until", 0.0)
    calls = []

    class Communicate:
        def __init__(self, text, voice):
            calls.append(text)

        async def stream(self):
            raise OSError("no network")
            yield  # pragma: no cover

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    assert speech.neural_wav("하나", "ko-KR-SunHiNeural") is None
    assert speech.neural_wav("둘", "ko-KR-SunHiNeural") is None
    assert calls == ["하나"]  # no second network wait within OFFLINE_RETRY_S
