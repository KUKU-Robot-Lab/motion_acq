from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from motion_acq.audio import PicoAudioRecorder
from motion_acq.feetech import zero_gripper_widths
from motion_acq.scripts import teleop_record
from motion_acq.scripts.teleop_record import (
    AsyncEpisodeCapture,
    AsyncLeRobotWriter,
    _CaptureSnapshot,
    _deferred_audio_frame,
)
from motion_acq.synchronization import SynchronizedGripperFrame


class _DatasetStub:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)

    def clear_episode_buffer(self) -> None:
        self.frames.clear()


def test_writer_flushes_deferred_frames_off_the_calling_thread() -> None:
    dataset = _DatasetStub()
    writer = AsyncLeRobotWriter(dataset, max_pending_frames=2, use_videos=False)
    try:
        writer.add_frame(lambda: {"value": 7})
        writer.flush()

        assert dataset.frames == [{"value": 7}]
        deferred_avg_ms, _, dataset_avg_ms, _ = writer.timing_stats()
        assert deferred_avg_ms >= 0.0
        assert dataset_avg_ms >= 0.0
    finally:
        writer.close()


def test_deferred_audio_alignment_can_use_a_packet_that_arrives_later(
    tmp_path: Path,
) -> None:
    recorder = PicoAudioRecorder(lambda: None, tmp_path)
    # The recorder continuously drains while idle, establishing the device to
    # workstation clock offset before an episode starts.
    recorder._accept_packet(
        {
            "data": b"\x00\x00" * 160,
            "sample_rate": 16_000,
            "channels": 1,
            "format": "pcm_s16le",
            "capture_timestamp_ns": 900_000_000,
            "sequence": 0,
        },
        receive_time_ns=905_000_000,
    )
    recorder.begin_episode(1_000_000_000)
    build = _deferred_audio_frame(
        {"row": 1},
        audio_recorder=recorder,
        target_time_ns=1_007_000_000,
        record_time_ns=1_020_000_000,
        stale_timeout_s=0.25,
        max_sync_skew_s=0.01,
    )
    recorder._accept_packet(
        {
            "data": b"\x01\x00" * 160,
            "sample_rate": 16_000,
            "channels": 1,
            "format": "pcm_s16le",
            "capture_timestamp_ns": 1_000_000_000,
            "sequence": 1,
        },
        receive_time_ns=1_025_000_000,
    )

    frame = build()

    assert frame["row"] == 1
    assert frame["observation.audio.healthy"].item() == 1


def test_episode_capture_keeps_robot_reads_off_control_thread(monkeypatch) -> None:
    dataset = _DatasetStub()
    writer = AsyncLeRobotWriter(dataset, max_pending_frames=2, use_videos=False)
    read_threads: list[str] = []

    class _RealEnv:
        def read(self, *, base_q):
            read_threads.append(threading.current_thread().name)
            return np.asarray(base_q, dtype=np.float32)

    monkeypatch.setattr(
        teleop_record,
        "canonicalize_command",
        lambda q, *, runtime, openings: np.asarray(q, dtype=np.float32).copy(),
    )
    monkeypatch.setattr(
        teleop_record,
        "read_camera_samples",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        teleop_record,
        "synchronized_gripper_frame",
        lambda *args, **kwargs: SynchronizedGripperFrame(
            widths=zero_gripper_widths(),
            frame={},
            healthy_for_gate=True,
        ),
    )
    capture = AsyncEpisodeCapture(
        real_env=_RealEnv(),
        runtime=object(),
        dataset_writer=writer,
        cameras=[],
        camera_names=[],
        camera_width=640,
        camera_height=480,
        camera_stale_timeout_s=0.25,
        grippers=None,
        gripper_stale_timeout_s=0.25,
        max_sync_skew_s=0.1,
        sensor_loss_timeout_s=1.0,
        task="test",
        audio_recorder=None,
    )
    try:
        for value in (1.0, 2.0):
            now_ns = time.monotonic_ns()
            capture.submit(
                _CaptureSnapshot(
                    base_q=np.array([value], dtype=np.float32),
                    action_q=np.array([value * 10], dtype=np.float32),
                    openings={"left": 0.0, "right": 0.0},
                    target_time_ns=now_ns,
                    record_time_ns=now_ns,
                    tracking_time_ns=now_ns,
                    submitted_s=time.perf_counter(),
                )
            )
        states, actions, n_frames = capture.finish()
        writer.flush()

        assert n_frames == 1
        np.testing.assert_array_equal(states, [[1.0]])
        np.testing.assert_array_equal(actions, [[20.0]])
        assert read_threads == ["handumi-episode-capture"] * 2
        assert len(dataset.frames) == 1
    finally:
        capture.close()
        writer.close()
