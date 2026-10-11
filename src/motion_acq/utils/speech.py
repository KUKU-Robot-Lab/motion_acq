"""Cross-platform text-to-speech for recording feedback.

Same approach LeRobot uses (``lerobot.utils.utils.say``/``log_say``): shell out
to the OS's built-in speech synthesizer, so recorders can announce episode
state hands-free (the collector is wearing the HandUMI shells, not looking at
a terminal). No extra Python dependency.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess

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


def say(
    text: str, *, blocking: bool = False, lang: str | None = None, interrupt: bool = False
) -> None:
    """Speak ``text`` using the OS's built-in TTS. No-op with a warning if
    the platform's speech command isn't available. ``lang`` (e.g. "ko")
    picks the voice language where the platform allows it (Linux);
    ``interrupt`` cuts what is still being spoken (Linux), for a cue that must
    be heard now ("start moving") rather than after a long prompt."""
    system = platform.system()
    if interrupt and system == "Linux":
        try:
            subprocess.run(["spd-say", "-C"], check=False, timeout=2.0, env=_session_env())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

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
