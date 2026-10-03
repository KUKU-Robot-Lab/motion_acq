"""LeRobot dataset read/write boundary for handumi."""

from typing import Any

from motion_acq.dataset.capture import (
    CAMERA_STALE_TIMEOUT_S,
    FEETECH_SAMPLE_HZ,
    GRIPPER_STALE_TIMEOUT_S,
    MAX_SYNC_SKEW_S,
    SENSOR_LOSS_TIMEOUT_S,
    SYNC_LAG_S,
    TRACKING_LOSS_TIMEOUT_S,
)
from motion_acq.dataset.quality import (
    EpisodeQualityConfig,
    EpisodeQualityReport,
    QualityFinding,
    validate_episode,
    write_quality_report,
)
from motion_acq.dataset.raw import (
    HANDUMI_RAW_IMAGE_KEYS,
    HANDUMI_RAW_STATE_NAMES,
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
    raw_state_feature,
    validate_raw_state_shape,
)
from motion_acq.dataset.reader import (
    DatasetDownloadResult,
    DatasetRef,
    RawEpisode,
    dataset_root_from_repo_id,
    download_dataset,
    ensure_dataset,
    ensure_metadata,
    handumi_metadata,
    load_raw_episode,
    load_raw_episode_states,
    open_dataset,
    recording_device,
    validate_raw_state_metadata,
)


def __getattr__(name: str) -> Any:
    """Lazily expose writer symbols without importing pandas at package import time."""
    writer_symbols = {
        "CHUNKS_SIZE",
        "EpisodeResult",
        "chunk_and_file",
        "info_path",
        "load_info",
        "update_handumi_metadata",
        "write_dataset",
    }
    if name in writer_symbols:
        from motion_acq.dataset import writer

        return getattr(writer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CHUNKS_SIZE",
    "CAMERA_STALE_TIMEOUT_S",
    "DatasetDownloadResult",
    "DatasetRef",
    "EpisodeResult",
    "EpisodeQualityConfig",
    "EpisodeQualityReport",
    "FEETECH_SAMPLE_HZ",
    "GRIPPER_STALE_TIMEOUT_S",
    "RawEpisode",
    "QualityFinding",
    "HANDUMI_RAW_IMAGE_KEYS",
    "HANDUMI_RAW_STATE_NAMES",
    "HANDUMI_RAW_STATE_SIZE",
    "LEFT_GRIPPER_INDEX",
    "LEFT_POSE_SLICE",
    "MAX_SYNC_SKEW_S",
    "RIGHT_GRIPPER_INDEX",
    "RIGHT_POSE_SLICE",
    "SENSOR_LOSS_TIMEOUT_S",
    "SYNC_LAG_S",
    "TRACKING_LOSS_TIMEOUT_S",
    "chunk_and_file",
    "dataset_root_from_repo_id",
    "download_dataset",
    "ensure_dataset",
    "ensure_metadata",
    "handumi_metadata",
    "info_path",
    "load_info",
    "load_raw_episode_states",
    "load_raw_episode",
    "open_dataset",
    "raw_state_feature",
    "recording_device",
    "update_handumi_metadata",
    "validate_raw_state_metadata",
    "validate_raw_state_shape",
    "validate_episode",
    "write_quality_report",
    "write_dataset",
]
