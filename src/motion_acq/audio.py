"""Episode- and frame-aligned PICO microphone capture for HandUMI datasets."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("motion_acq.audio")

AUDIO_KEY = "observation.audio"
AUDIO_PATH = "audio/{audio_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.wav"
PCM_SAMPLE_WIDTH_BYTES = 2
MISSING_AUDIO_SAMPLE_RATE = 16_000
MISSING_AUDIO_CHANNELS = 1


def _scalar_int(value: int | bool) -> Any:
    # Keep NumPy out of the capture thread's hot path and import it only when
    # a recorder row is actually built.
    import numpy as np

    return np.array([int(value)], dtype=np.int64)


@dataclass(frozen=True)
class _AudioChunk:
    data: bytes
    device_time_ns: int
    receive_time_ns: int
    aligned_time_ns: int
    sequence: int


class AudioCaptureError(RuntimeError):
    """The headset audio stream could not produce a valid episode."""


class PicoAudioRecorder:
    """Continuously drain XRoboToolkit audio and stage one PCM WAV per episode.

    Draining while idle prevents the SDK's bounded queue from overflowing and
    ensures a new episode never begins with audio left over from the wait state.
    """

    def __init__(self, xrt_getter: Callable[[], Any], root: Path) -> None:
        self._xrt_getter = xrt_getter
        self._root = Path(root)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recording = False
        self._chunks: list[_AudioChunk] = []
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._format: str | None = None
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._sequence_gaps = 0
        self._error: BaseException | None = None
        self._prepared_path: Path | None = None
        self._prepared_info: dict[str, object] | None = None
        self._clock_offset_ns: int | None = None
        self._clock_offsets_ns: deque[int] = deque(maxlen=200)
        self._episode_clock_offset_ns: int | None = None
        self._episode_start_ns: int | None = None
        self._frame_targets_ns: list[int] = []

    def start(self) -> None:
        xrt = self._xrt_getter()
        if xrt is None or not callable(getattr(xrt, "get_audio_frame", None)):
            raise AudioCaptureError(
                "The installed xrobotoolkit_sdk has no get_audio_frame API. "
                "Rebuild/install the audio-enabled SDK from external_dependencies."
            )
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-pico-audio",
            daemon=True,
        )
        self._thread.start()
        log.info("PICO audio capture ready; enable Audio in the headset app.")

    def begin_episode(self, start_time_ns: int | None = None) -> None:
        with self._lock:
            self._raise_if_failed_locked()
            self._remove_prepared_locked()
            self._reset_episode_locked()
            self._episode_start_ns = int(
                time.monotonic_ns() if start_time_ns is None else start_time_ns
            )
            self._episode_clock_offset_ns = self._clock_offset_ns
            self._recording = True

    def synchronized_frame(
        self,
        target_time_ns: int,
        record_time_ns: int,
        *,
        stale_timeout_s: float,
        max_sync_skew_s: float,
    ) -> tuple[dict[str, Any], bool]:
        """Describe the PCM sample nearest one dataset-row target time.

        Audio device timestamps are translated into the workstation monotonic
        domain using the lowest observed one-way receive offset. The WAV is
        later rendered on the same episode clock, so ``sample_index`` points
        into the committed file.
        """
        with self._lock:
            self._note_frame_target_locked(int(target_time_ns))
            sample_rate = self._sample_rate
            channels = self._channels
            episode_start_ns = self._episode_start_ns
            nearest: _AudioChunk | None = None
            nearest_time_ns = 0
            nearest_error_ns = 2**63 - 1
            if sample_rate and channels:
                # Chunks arrive in sequence order. Searching backwards normally
                # examines only the handful around the recorder's sync lag.
                for chunk in reversed(self._chunks):
                    chunk_time_ns = self._aligned_chunk_time_ns(
                        chunk,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                    chunk_frames = len(chunk.data) // (
                        PCM_SAMPLE_WIDTH_BYTES * channels
                    )
                    chunk_end_ns = chunk_time_ns + int(
                        chunk_frames * 1e9 / sample_rate
                    )
                    candidate_time_ns = min(
                        max(target_time_ns, chunk_time_ns), chunk_end_ns
                    )
                    error_ns = abs(candidate_time_ns - target_time_ns)
                    if error_ns <= nearest_error_ns:
                        nearest = chunk
                        nearest_time_ns = candidate_time_ns
                        nearest_error_ns = error_ns
                    if chunk_end_ns < target_time_ns:
                        break

        sample_index = 0
        if sample_rate and channels and episode_start_ns is not None:
            sample_index = max(
                0,
                int(round((target_time_ns - episode_start_ns) * sample_rate / 1e9)),
            )

        age_ns = (
            2**63 - 1
            if nearest is None
            else max(0, int(record_time_ns) - nearest.receive_time_ns)
        )
        healthy = bool(
            nearest is not None
            and nearest_error_ns <= int(max_sync_skew_s * 1e9)
            and age_ns <= int(stale_timeout_s * 1e9)
        )
        frame = {
            "observation.audio.device_time_ns": _scalar_int(
                nearest.device_time_ns if nearest is not None else 0
            ),
            "observation.audio.pc_monotonic_ns": _scalar_int(
                nearest.receive_time_ns if nearest is not None else 0
            ),
            "observation.audio.aligned_time_ns": _scalar_int(nearest_time_ns),
            "observation.audio.sample_index": _scalar_int(sample_index),
            "observation.audio.sample_rate": _scalar_int(sample_rate or 0),
            "observation.audio.sequence": _scalar_int(
                nearest.sequence if nearest is not None else 0
            ),
            "observation.audio.healthy": _scalar_int(healthy),
        }
        return frame, healthy

    def prepare_episode(self, expected_frames: int | None = None) -> dict[str, object]:
        """Stop the active episode and create a staged WAV without publishing it."""
        with self._lock:
            self._recording = False
            self._raise_if_failed_locked()
            chunks = list(self._chunks)
            sample_rate = self._sample_rate
            channels = self._channels
            audio_format = self._format
            first_timestamp_ns = self._first_timestamp_ns
            last_timestamp_ns = self._last_timestamp_ns
            first_sequence = self._first_sequence
            last_sequence = self._last_sequence
            sequence_gaps = self._sequence_gaps
            episode_start_ns = self._episode_start_ns
            frame_targets_ns = list(self._frame_targets_ns)
            clock_offset_ns = self._episode_clock_offset_ns

        if expected_frames is not None:
            if expected_frames <= 0 or expected_frames > len(frame_targets_ns):
                raise AudioCaptureError(
                    "Audio frame alignment does not match the episode frame count "
                    f"({len(frame_targets_ns)} timing rows for {expected_frames} frames)."
                )
            frame_targets_ns = frame_targets_ns[:expected_frames]
        first_frame_target_ns = frame_targets_ns[0] if frame_targets_ns else None
        last_frame_target_ns = frame_targets_ns[-1] if frame_targets_ns else None
        frame_period_ns = (
            frame_targets_ns[-1] - frame_targets_ns[-2]
            if len(frame_targets_ns) >= 2
            else None
        )

        missing_audio = not chunks or sample_rate is None or channels is None
        if missing_audio:
            # An operator save gesture must still commit the robot episode when
            # the headset audio toggle was left off or its stream disappeared.
            # Frame-level audio health already records this condition as 0, so
            # retain the demonstration with an aligned silent WAV instead of
            # silently turning RIGHT/save into a reset.  Calls without frame
            # timing keep the strict behavior used by standalone audio capture.
            if expected_frames is None or not frame_targets_ns:
                raise AudioCaptureError(
                    "No PICO audio arrived during the episode. "
                    "Enable Audio in the headset app."
                )
            sample_rate = MISSING_AUDIO_SAMPLE_RATE
            channels = MISSING_AUDIO_CHANNELS
            audio_format = "pcm_s16le"
            log.warning(
                "No PICO audio arrived during the episode; saving aligned "
                "silence so the robot episode can still be committed. Enable "
                "Audio in the headset app for microphone recording."
            )
        staging_dir = self._root / ".audio-staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix="episode-", suffix=".wav", dir=staging_dir)
        os.close(fd)
        staged = Path(raw_path)
        try:
            with wave.open(str(staged), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
                wav.setframerate(sample_rate)
                if first_frame_target_ns is None or episode_start_ns is None:
                    for chunk in chunks:
                        wav.writeframesraw(chunk.data)
                else:
                    end_ns = int(last_frame_target_ns or first_frame_target_ns) + int(
                        frame_period_ns or round(1e9 / sample_rate)
                    )
                    rendered = self._render_aligned_pcm(
                        chunks,
                        sample_rate=sample_rate,
                        channels=channels,
                        start_ns=episode_start_ns,
                        end_ns=end_ns,
                        offset_ns=clock_offset_ns,
                    )
                    wav.writeframesraw(rendered)
        except Exception:
            staged.unlink(missing_ok=True)
            raise

        with wave.open(str(staged), "rb") as wav:
            sample_frames = wav.getnframes()
        info: dict[str, object] = {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": PCM_SAMPLE_WIDTH_BYTES,
            "format": audio_format or "pcm_s16le",
            "sample_frames": sample_frames,
            "duration_s": sample_frames / sample_rate,
            "first_capture_timestamp_ns": first_timestamp_ns,
            "last_capture_timestamp_ns": last_timestamp_ns,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "sequence_gaps": sequence_gaps,
            "episode_start_time_ns": episode_start_ns,
            "first_frame_target_time_ns": first_frame_target_ns,
            "last_frame_target_time_ns": last_frame_target_ns,
            "clock_offset_ns": clock_offset_ns,
            "synthetic_silence": missing_audio,
        }
        with self._lock:
            self._prepared_path = staged
            self._prepared_info = info
        if sequence_gaps:
            log.warning("PICO audio episode contains %d missing chunk(s).", sequence_gaps)
        return dict(info)

    def commit_episode(self, episode_index: int, *, chunks_size: int = 1000) -> Path:
        with self._lock:
            if self._prepared_path is None:
                raise AudioCaptureError("Audio episode was not prepared before commit.")
            staged = self._prepared_path
            self._prepared_path = None
            self._prepared_info = None
        chunk_index, file_index = divmod(int(episode_index), int(chunks_size))
        relative = Path(
            AUDIO_PATH.format(
                audio_key=AUDIO_KEY,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        destination = self._root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            staged.unlink(missing_ok=True)
            raise AudioCaptureError(f"Refusing to overwrite existing audio: {destination}")
        os.replace(staged, destination)
        self._remove_empty_staging_dir()
        return destination

    def cancel_episode(self) -> None:
        with self._lock:
            self._recording = False
            self._remove_prepared_locked()
            self._reset_episode_locked()
        self._remove_empty_staging_dir()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.cancel_episode()

    def _run(self) -> None:
        while not self._stop.is_set():
            xrt = self._xrt_getter()
            if xrt is None:
                self._stop.wait(0.05)
                continue
            try:
                packet = xrt.get_audio_frame(timeout_ms=100, latest=False)
            except Exception as exc:  # noqa: BLE001 - SDK can fail during recovery.
                if self._stop.is_set():
                    return
                log.warning("PICO audio read failed; retrying: %s", exc)
                time.sleep(0.1)
                continue
            if packet is None:
                continue
            try:
                self._accept_packet(packet, receive_time_ns=time.monotonic_ns())
            except Exception as exc:  # noqa: BLE001 - validate all native payload failures.
                with self._lock:
                    self._error = exc
                    self._recording = False
                log.error("PICO audio capture failed: %s", exc)

    def _accept_packet(self, packet: Any, *, receive_time_ns: int | None = None) -> None:
        data = bytes(packet["data"])
        sample_rate = int(packet["sample_rate"])
        channels = int(packet["channels"])
        audio_format = str(packet.get("format") or "pcm_s16le").lower()
        timestamp_ns = int(packet.get("capture_timestamp_ns", 0))
        sequence = int(packet.get("sequence", 0))
        received_ns = int(
            time.monotonic_ns() if receive_time_ns is None else receive_time_ns
        )
        if sample_rate <= 0 or channels <= 0 or not data:
            raise AudioCaptureError("XRoboToolkit returned an invalid audio chunk.")
        if audio_format not in {"pcm16", "pcm_s16le", "s16le"}:
            raise AudioCaptureError(f"Unsupported XRoboToolkit audio format: {audio_format}")
        if len(data) % (PCM_SAMPLE_WIDTH_BYTES * channels):
            raise AudioCaptureError("XRoboToolkit audio chunk is not whole PCM16 frames.")
        with self._lock:
            if timestamp_ns > 0:
                observed_offset_ns = received_ns - timestamp_ns
                self._clock_offsets_ns.append(observed_offset_ns)
                self._clock_offset_ns = min(self._clock_offsets_ns)
            if not self._recording:
                return
            if self._sample_rate is None:
                self._sample_rate = sample_rate
                self._channels = channels
                self._format = audio_format
                self._first_timestamp_ns = timestamp_ns
                self._first_sequence = sequence
            elif sample_rate != self._sample_rate or channels != self._channels:
                raise AudioCaptureError(
                    "PICO audio format changed within an episode "
                    f"({self._sample_rate} Hz/{self._channels} ch -> "
                    f"{sample_rate} Hz/{channels} ch)."
                )
            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self._sequence_gaps += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            self._last_timestamp_ns = timestamp_ns
            aligned_time_ns = (
                timestamp_ns + self._episode_clock_offset_ns
                if timestamp_ns > 0 and self._episode_clock_offset_ns is not None
                else 0
            )
            if timestamp_ns > 0 and self._episode_clock_offset_ns is None:
                self._episode_clock_offset_ns = self._clock_offset_ns
                aligned_time_ns = timestamp_ns + int(self._episode_clock_offset_ns or 0)
            self._chunks.append(
                _AudioChunk(
                    data,
                    timestamp_ns,
                    received_ns,
                    aligned_time_ns,
                    sequence,
                )
            )

    @staticmethod
    def _aligned_chunk_time_ns(
        chunk: _AudioChunk,
        *,
        sample_rate: int,
        channels: int,
        offset_ns: int | None = None,
    ) -> int:
        if chunk.aligned_time_ns > 0:
            return chunk.aligned_time_ns
        if chunk.device_time_ns > 0 and offset_ns is not None:
            return chunk.device_time_ns + offset_ns
        chunk_frames = len(chunk.data) // (PCM_SAMPLE_WIDTH_BYTES * channels)
        return chunk.receive_time_ns - int(chunk_frames * 1e9 / sample_rate)

    @classmethod
    def _render_aligned_pcm(
        cls,
        chunks: list[_AudioChunk],
        *,
        sample_rate: int,
        channels: int,
        start_ns: int,
        end_ns: int,
        offset_ns: int | None,
    ) -> bytes:
        frame_bytes = PCM_SAMPLE_WIDTH_BYTES * channels
        output_frames = max(1, int(round((end_ns - start_ns) * sample_rate / 1e9)))
        output = bytearray(output_frames * frame_bytes)
        for chunk in chunks:
            chunk_start_ns = cls._aligned_chunk_time_ns(
                chunk,
                sample_rate=sample_rate,
                channels=channels,
                offset_ns=offset_ns,
            )
            destination_frame = int(round((chunk_start_ns - start_ns) * sample_rate / 1e9))
            source_frame = max(0, -destination_frame)
            destination_frame = max(0, destination_frame)
            chunk_frames = len(chunk.data) // frame_bytes
            copy_frames = min(chunk_frames - source_frame, output_frames - destination_frame)
            if copy_frames <= 0:
                continue
            source_start = source_frame * frame_bytes
            destination_start = destination_frame * frame_bytes
            output[destination_start : destination_start + copy_frames * frame_bytes] = (
                chunk.data[source_start : source_start + copy_frames * frame_bytes]
            )
        return bytes(output)

    def _note_frame_target_locked(self, target_time_ns: int) -> None:
        self._frame_targets_ns.append(target_time_ns)

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            error = self._error
            self._error = None
            raise AudioCaptureError(str(error)) from error

    def _reset_episode_locked(self) -> None:
        self._chunks = []
        self._sample_rate = None
        self._channels = None
        self._format = None
        self._first_timestamp_ns = None
        self._last_timestamp_ns = None
        self._first_sequence = None
        self._last_sequence = None
        self._sequence_gaps = 0
        self._episode_start_ns = None
        self._episode_clock_offset_ns = None
        self._frame_targets_ns = []

    def _remove_prepared_locked(self) -> None:
        if self._prepared_path is not None:
            self._prepared_path.unlink(missing_ok=True)
        self._prepared_path = None
        self._prepared_info = None

    def _remove_empty_staging_dir(self) -> None:
        staging_dir = self._root / ".audio-staging"
        try:
            staging_dir.rmdir()
        except OSError:
            pass


def audio_metadata(enabled: bool) -> dict[str, object]:
    return {
        "enabled": bool(enabled),
        "key": AUDIO_KEY,
        "path": AUDIO_PATH,
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_width_bytes": PCM_SAMPLE_WIDTH_BYTES,
        "episode_aligned": True,
        "frame_aligned": True,
        "timestamp_clock": "pc_monotonic",
        "sample_index_origin": "episode_start_time_ns",
    }


def audio_features() -> dict[str, dict[str, object]]:
    """Per-row audio timing and health fields for frame-level alignment."""
    return {
        f"observation.audio.{key}": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
        for key in (
            "device_time_ns",
            "pc_monotonic_ns",
            "aligned_time_ns",
            "sample_index",
            "sample_rate",
            "sequence",
            "healthy",
        )
    }


def validate_audio_files(root: Path, total_episodes: int, *, chunks_size: int = 1000) -> None:
    """Require one readable non-empty WAV for every committed episode."""
    for episode_index in range(total_episodes):
        chunk_index, file_index = divmod(episode_index, chunks_size)
        path = Path(root) / AUDIO_PATH.format(
            audio_key=AUDIO_KEY,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        if not path.is_file():
            raise RuntimeError(f"Dataset is missing episode audio: {path}.")
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getnframes() <= 0 or wav.getframerate() <= 0 or wav.getnchannels() <= 0:
                    raise RuntimeError(f"Dataset has empty or invalid episode audio: {path}.")
        except (OSError, EOFError, wave.Error) as exc:
            raise RuntimeError(f"Dataset has invalid episode audio: {path}.") from exc
