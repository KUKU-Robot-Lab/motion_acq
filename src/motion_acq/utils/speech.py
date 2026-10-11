"""Cross-platform text-to-speech for recording feedback.

Same approach LeRobot uses (``lerobot.utils.utils.say``/``log_say``): shell out
to the OS's built-in speech synthesizer, so recorders can announce episode
state hands-free (the collector is wearing the HandUMI shells, not looking at
a terminal). No extra Python dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import queue
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

log = logging.getLogger("handumi.speech")


def _session_env() -> dict[str, str]:
    """The environment with the login session's runtime dir (10.11).

    A process started over ssh has no XDG_RUNTIME_DIR, so spd-say would spawn
    its own speech-dispatcher that cannot reach the desktop's audio; point it
    at the user's /run/user/<uid>, where the desktop's dispatcher listens.
    """
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        runtime = f"/run/user/{os.getuid()}"
        if os.path.isdir(runtime):
            env["XDG_RUNTIME_DIR"] = runtime
    return env


# ---- neural voice (10.12 user: espeak's Korean is hard to understand) -------------------------
# Microsoft's neural voices through edge-tts (free, needs the internet the first time a sentence
# is said); each sentence is kept as a WAV in CACHE_DIR, so known prompts play at once and offline.
# MACQ_TTS=espeak forces the old voice; MACQ_TTS_VOICE picks another edge-tts voice.
NEURAL_VOICES = {"ko": "ko-KR-SunHiNeural"}
CACHE_DIR = Path(os.environ.get("MACQ_TTS_CACHE", Path.home() / ".cache" / "macq" / "tts"))
SAMPLE_RATE = 24000
SYNTH_TIMEOUT_S = 6.0
OFFLINE_RETRY_S = 60.0  # after a failed synthesis, use espeak for this long (no repeated waits)


def neural_voice(lang: str | None) -> str | None:
    if os.environ.get("MACQ_TTS", "").lower() == "espeak" or platform.system() != "Linux":
        return None
    voice = NEURAL_VOICES.get(lang or "")
    return os.environ.get("MACQ_TTS_VOICE", voice) if voice else None


def cache_path(text: str, voice: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(f'{voice}|{text}'.encode()).hexdigest()[:20]}.wav"


_offline_until = 0.0


def neural_wav(text: str, voice: str) -> Path | None:
    """The cached WAV of ``text`` in ``voice``, made now if needed; None when it cannot be made."""
    global _offline_until
    path = cache_path(text, voice)
    if path.exists():
        return path
    if time.monotonic() < _offline_until:
        return None
    try:
        import asyncio

        import edge_tts
        import miniaudio
    except ImportError:
        _offline_until = float("inf")
        log.warning("edge-tts / miniaudio not installed (uv sync --extra voice); using espeak.")
        return None

    async def fetch() -> bytes:
        data = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                data += chunk["data"]
        return bytes(data)

    result: dict = {}

    def run() -> None:  # own thread and loop: say() may be called from a thread with a running loop
        try:
            result["mp3"] = asyncio.run(asyncio.wait_for(fetch(), SYNTH_TIMEOUT_S))
        except Exception as exc:  # noqa: BLE001 - network, service or timeout: fall back
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True, name="tts-synth")
    worker.start()
    worker.join(SYNTH_TIMEOUT_S + 1.0)
    mp3 = result.get("mp3")
    if not mp3:
        _offline_until = time.monotonic() + OFFLINE_RETRY_S
        log.warning("Neural voice failed (%s); espeak for %.0f s.", result.get("error", "timeout"), OFFLINE_RETRY_S)
        return None
    pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                           sample_rate=SAMPLE_RATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.part")
    with wave.open(str(partial), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.samples.tobytes())
    partial.replace(path)
    return path


def prefetch(texts, lang: str = "ko") -> int:
    """Make the WAVs of known prompts ahead (a countdown must not wait for the network). How many exist."""
    voice = neural_voice(lang)
    if voice is None:
        return 0
    return sum(neural_wav(text, voice) is not None for text in texts)


class _Player:
    """One queue of sentences played in order (like speech-dispatcher); interrupt clears it."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._current: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def speak(self, text: str, voice: str, lang: str | None, *, blocking: bool, interrupt: bool) -> None:
        if interrupt:
            self.interrupt()
        done = threading.Event()
        self._queue.put((text, voice, lang, done))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="tts-player")
                self._thread.start()
        if blocking:
            done.wait()

    def interrupt(self) -> None:
        try:
            while True:
                self._queue.get_nowait()[3].set()
        except queue.Empty:
            pass
        with self._lock:
            if self._current is not None and self._current.poll() is None:
                self._current.terminate()

    def _run(self) -> None:
        while True:
            text, voice, lang, done = self._queue.get()
            try:
                path = neural_wav(text, voice)
                if path is None:
                    _espeak(text, lang=lang, blocking=True)
                    continue
                with self._lock:
                    self._current = subprocess.Popen(["paplay", str(path)], env=_session_env(),
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    current = self._current
                current.wait()
            except FileNotFoundError:
                _espeak(text, lang=lang, blocking=True)
            except Exception as exc:  # noqa: BLE001 - speech must never stop the caller
                log.warning("Voice failed: %s", exc)
            finally:
                done.set()


_player = _Player()


def say(
    text: str, *, blocking: bool = False, lang: str | None = None, interrupt: bool = False
) -> None:
    """Speak ``text``. Korean (``lang="ko"``) uses the cached neural voice when it can,
    else the OS's built-in TTS (no-op with a warning if missing). ``interrupt``
    cuts what is still being spoken (Linux), for a cue that must be heard now
    ("start moving") rather than after a long prompt."""
    voice = neural_voice(lang)
    if voice is not None:
        if interrupt:
            _espeak_cancel()
        _player.speak(text, voice, lang, blocking=blocking, interrupt=interrupt)
        return
    if interrupt:
        _player.interrupt()
    _espeak(text, lang=lang, blocking=blocking, interrupt=interrupt)


def _espeak_cancel() -> None:
    if platform.system() == "Linux":
        try:
            subprocess.run(["spd-say", "-C"], check=False, timeout=2.0, env=_session_env())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


def _espeak(text: str, *, blocking: bool = False, lang: str | None = None, interrupt: bool = False) -> None:
    """The OS's built-in speech synthesizer."""
    system = platform.system()
    if interrupt:
        _espeak_cancel()

    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text] if lang is None else ["spd-say", "-l", lang, text]
        if blocking:
            cmd.append("--wait")
    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
    else:
        log.warning("Unsupported OS for text-to-speech (%s); skipping.", system)
        return

    env = _session_env() if system == "Linux" else None
    try:
        if blocking:
            subprocess.run(cmd, check=True, env=env)
        else:
            subprocess.Popen(
                cmd, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0,
            )
    except FileNotFoundError:
        log.warning(
            "Text-to-speech command %r not found; install it (e.g. `spd-say` "
            "ships with speech-dispatcher on Linux) or pass --no-sounds.",
            cmd[0],
        )


_STEP = re.compile(r"\[([^\]\d]+?) (\d+)/\d+\]")
_PAREN = re.compile(r"\s*\([^()]*\)")


def speech_text(text: str) -> str:
    """The part of an on-screen Korean prompt worth hearing (10.11 user).

    Drops the details in parentheses (numbers, sensor names), reads a step
    tag "[오른손 3/11]" as "오른손 3번", and spells out symbols a voice
    reads badly.
    """
    out = _STEP.sub(r"\1 \2번.", text)
    while _PAREN.search(out):
        out = _PAREN.sub("", out)
    out = out.replace("Enter 를", "엔터를").replace("Enter", "엔터")
    out = out.replace("·", ", ").replace("[", "").replace("]", "")
    out = re.sub(r"(?<=\w)-(?=\w)", " ", out)
    return " ".join(out.split()).strip(" :")


class Voice:
    """Read on-screen prompts aloud (``speech_text`` of each), or stay quiet.

    A prompt holding one of ``now_cues`` cuts whatever is still being read, so
    a cue that starts a recording is heard at once.
    """

    def __init__(self, enabled: bool, *, lang: str | None = None, now_cues: tuple[str, ...] = ()) -> None:
        self.enabled = enabled
        self.lang = lang
        self.now_cues = now_cues

    def __call__(self, text: str) -> None:
        line = speech_text(text)
        if self.enabled and line:
            say(line, lang=self.lang, interrupt=any(cue in text for cue in self.now_cues))


def log_say(text: str, *, play_sounds: bool = True, blocking: bool = False) -> None:
    """Log ``text`` and, if ``play_sounds``, also speak it."""
    log.info(text)
    if play_sounds:
        say(text, blocking=blocking)
