"""Hands-free episode control by spoken command.

The recorder's other hands-free control is a double gripper squeeze
(:class:`~motion_acq.tracking.gestures.DoubleClapDetector`), which needs the
collector to interrupt whatever the shells are holding. Speaking is more
ergonomic, so voice is the recorder's default control and the squeeze is the
opt-in fallback for noisy rooms.

Recognition is offline (Vosk) with a *closed grammar*: the recognizer can only
ever return the three control phrases or ``[unk]``, so background conversation
cannot be mistaken for a command. Audio comes from the OS default input, which
is what follows a headset in and out on PipeWire/PulseAudio.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from urllib.request import urlopen

log = logging.getLogger("handumi.voice")

SAMPLE_RATE = 16000
_BLOCK_FRAMES = 4000  # 0.25s per recognizer feed

# Phrase -> action. Vosk is given exactly these phrases plus "[unk]", so a
# spoken sentence outside the grammar decodes to "[unk]" and is dropped.
COMMAND_PHRASES: dict[str, str] = {
    "start recording": "start",
    "stop recording": "stop",
    "restart": "restart",
}

_UNKNOWN_TOKEN = "[unk]"

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
_MODEL_ENV = "HANDUMI_VOSK_MODEL"

_MISSING_DEPS = (
    "Voice control needs the `vosk` and `sounddevice` packages.\n"
    "  Install them:  uv pip install vosk sounddevice\n"
    "  Or record without voice:  --no-voice-control (add --clap-control for "
    "hands-free gripper squeezes)."
)


class VoiceUnavailableError(RuntimeError):
    """Voice control was requested but cannot run on this machine."""


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "handumi" / "vosk"


def resolve_model_path(*, download: bool = True) -> Path:
    """Return the local Vosk model directory, fetching it once if needed.

    ``HANDUMI_VOSK_MODEL`` overrides the location for air-gapped machines or
    for trying a larger model.
    """
    override = os.environ.get(_MODEL_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise VoiceUnavailableError(
                f"{_MODEL_ENV} points at {path}, which is not a directory."
            )
        return path

    path = _cache_root() / MODEL_NAME
    if path.is_dir():
        return path
    if not download:
        raise VoiceUnavailableError(f"Vosk model not downloaded yet (expected at {path}).")

    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / f"{MODEL_NAME}.zip"
    log.info("Downloading the speech model once (~40 MB) to %s ...", path.parent)
    try:
        with urlopen(MODEL_URL, timeout=60) as response, archive.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(path.parent)
    except Exception as exc:  # network, disk, corrupt archive
        raise VoiceUnavailableError(
            f"Could not download the speech model from {MODEL_URL}: {exc}\n"
            f"  Download it manually, unzip it, and set {_MODEL_ENV} to the folder."
        ) from exc
    finally:
        archive.unlink(missing_ok=True)

    if not path.is_dir():
        raise VoiceUnavailableError(f"Speech model archive did not contain {MODEL_NAME}.")
    return path


def _resolve_device(device: str | int | None):
    """Map ``device`` (index, name substring, or None) to a sounddevice id."""
    import sounddevice as sd

    if device is None:
        return None
    if isinstance(device, int) or str(device).isdigit():
        return int(device)
    needle = str(device).lower()
    matches = [
        index
        for index, info in enumerate(sd.query_devices())
        if info["max_input_channels"] > 0 and needle in info["name"].lower()
    ]
    if not matches:
        raise VoiceUnavailableError(f"No input device matching {device!r}.")
    return matches[0]


def describe_input_device(device: str | int | None = None) -> str:
    """Human-readable name of the microphone voice control would use."""
    import sounddevice as sd

    resolved = _resolve_device(device)
    if resolved is None:
        resolved = sd.default.device[0]
        if resolved is None or resolved < 0:
            raise VoiceUnavailableError("No default input device.")
    return str(sd.query_devices(resolved)["name"])


class VoiceCommandListener:
    """Background microphone listener exposing a non-blocking command poll.

    The recorder's loop runs to a per-frame time budget, so recognition happens
    on its own thread and :meth:`poll` only pops an already-decoded command.
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        confidence: float = 0.7,
        debounce_s: float = 1.5,
    ) -> None:
        self._device = device
        self._confidence = confidence
        self._debounce_s = debounce_s
        self._commands: deque[str] = deque(maxlen=4)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._muted_until = 0.0
        self._last_command: tuple[str, float] | None = None
        self.device_name = "unknown"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the model and microphone. Raises :class:`VoiceUnavailableError`."""
        try:
            import sounddevice  # noqa: F401
            import vosk
        except ImportError as exc:
            raise VoiceUnavailableError(_MISSING_DEPS) from exc

        vosk.SetLogLevel(-1)  # its default chatter would bury the recorder's logs
        model_path = resolve_model_path()
        try:
            self._model = vosk.Model(str(model_path))
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Could not load the speech model at {model_path}: {exc}"
            ) from exc

        try:
            self.device_name = describe_input_device(self._device)
        except Exception as exc:
            raise VoiceUnavailableError(f"No usable microphone: {exc}") from exc

        # Fail fast here rather than inside the worker: a missing mic must be a
        # startup error, not a silently dead control path mid-session.
        self._open_stream().close()

        self._thread = threading.Thread(
            target=self._run, name="handumi_voice", daemon=True
        )
        self._thread.start()
        log.info(
            "Voice control listening on %r (say: %s).",
            self.device_name,
            ", ".join(f'"{phrase}"' for phrase in COMMAND_PHRASES),
        )

    def stop(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- recorder-facing API ----------------------------------------------

    def poll(self) -> str | None:
        """Pop the oldest pending command (``start``/``stop``/``restart``)."""
        with self._lock:
            return self._commands.popleft() if self._commands else None

    def drain(self) -> None:
        """Discard pending commands, e.g. right after acting on one."""
        with self._lock:
            self._commands.clear()

    def mute(self, seconds: float) -> None:
        """Ignore audio for ``seconds``.

        The recorder announces state out loud ("Stop recording", "Restart
        recording"); without muting, its own text-to-speech would be heard as
        the very command it just executed.
        """
        self._muted_until = max(self._muted_until, time.monotonic() + seconds)
        self.drain()

    # -- internals ---------------------------------------------------------

    def _open_stream(self):
        import sounddevice as sd

        return sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=_BLOCK_FRAMES,
            device=_resolve_device(self._device),
            dtype="int16",
            channels=1,
        )

    def _grammar(self) -> str:
        return json.dumps([*COMMAND_PHRASES, _UNKNOWN_TOKEN])

    def _run(self) -> None:
        import vosk

        while not self._closed.is_set():
            try:
                recognizer = vosk.KaldiRecognizer(self._model, SAMPLE_RATE, self._grammar())
                recognizer.SetWords(True)
                with self._open_stream() as stream:
                    while not self._closed.is_set():
                        data, _overflowed = stream.read(_BLOCK_FRAMES)
                        if time.monotonic() < self._muted_until:
                            recognizer.Reset()
                            continue
                        if recognizer.AcceptWaveform(bytes(data)):
                            self._handle_result(recognizer.Result())
            except Exception as exc:
                if self._closed.is_set():
                    return
                # A headset unplugged mid-session kills the stream. Recording
                # must survive it, so reopen against whatever is default now.
                log.warning("Voice input dropped (%s); reopening microphone ...", exc)
                if self._closed.wait(1.0):
                    return

    def _handle_result(self, raw: str) -> None:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return
        # Breath, room noise, and clipped syllables decode to "[unk]" tokens
        # around the phrase ("restart" commonly arrives as "[unk] restart"),
        # so match on the phrase with those tokens removed. Anything else in
        # the utterance is itself "[unk]", so a command buried in a longer
        # sentence still cannot be assembled out of unrelated speech.
        text = " ".join(
            word
            for word in str(result.get("text", "")).split()
            if word != _UNKNOWN_TOKEN
        )
        action = COMMAND_PHRASES.get(text)
        if action is None:
            return

        words = [w for w in (result.get("result") or []) if w.get("word") != _UNKNOWN_TOKEN]
        confidences = [float(w.get("conf", 1.0)) for w in words]
        score = min(confidences) if confidences else 1.0
        if score < self._confidence:
            log.debug("Ignoring %r: confidence %.2f below threshold.", text, score)
            return

        now = time.monotonic()
        if self._last_command is not None:
            last_action, last_t = self._last_command
            if last_action == action and now - last_t < self._debounce_s:
                return
        self._last_command = (action, now)

        log.info("Voice command: %r -> %s", text, action)
        with self._lock:
            self._commands.append(action)


def speech_duration_s(text: str) -> float:
    """Rough time the OS text-to-speech needs for ``text``, for muting."""
    return 0.6 + 0.38 * max(1, len(text.split()))
