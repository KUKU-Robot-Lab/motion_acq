"""Tests never reach the network for speech: the neural voice is off unless a test turns it on."""

import pytest


@pytest.fixture(autouse=True)
def _no_neural_voice(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("MACQ_TTS", "espeak")
    from motion_acq.utils import speech

    monkeypatch.setattr(speech, "CACHE_DIR", tmp_path_factory.mktemp("tts"))
