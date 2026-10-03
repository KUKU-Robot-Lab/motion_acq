"""USB camera setup helpers and frame collection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from motion_acq.cameras.base import CameraDevice, CameraSample
from motion_acq.cameras.opencv import OpenCVCameraDevice
from motion_acq.cameras.zedmini import ZedMiniCameraDevice
from motion_acq.config import load_rig_section

log = logging.getLogger("handumi.record")

CameraSpec = dict[str, Any]


def build_camera_specs(
    cam_ids: list[int | str],
    *,
    camera_names: Sequence[str] | None = None,
    laptop_camera: bool,
    laptop_cam_id: int,
    laptop_cam_name: str,
    rig_config: Path | None = None,
    default_fps: int = 30,
    default_width: int = 640,
    default_height: int = 480,
) -> tuple[list[CameraSpec], str | None]:
    if camera_names is None:
        names = ["left_wrist", "right_wrist"]
        names.extend(f"cam_{i}" for i in range(2, len(cam_ids)))
    else:
        names = list(camera_names)
    if len(names) != len(cam_ids):
        raise ValueError(
            f"Expected {len(names)} camera IDs for {names}, got {len(cam_ids)}."
        )
    rig_cameras = (
        load_rig_section(rig_config, "cameras") if rig_config is not None else {}
    )
    specs = []
    for i, cam_id in enumerate(cam_ids):
        name = names[i] if i < len(names) else f"cam_{i}"
        entry = rig_cameras.get(name) or {}
        if not isinstance(entry, dict):
            raise SystemExit(f"Invalid cameras.{name} in {rig_config}; expected a mapping.")
        camera_type = _normalize_camera_type(entry.get("type", "opencv"))
        spec: CameraSpec = {"id": cam_id, "name": name, "is_laptop": False}
        if rig_config is not None:
            spec.update(
                {
                    "type": camera_type,
                    "width": _positive_camera_int(
                        entry, "width", name, default_width
                    ),
                    "height": _positive_camera_int(
                        entry, "height", name, default_height
                    ),
                    "fps": _positive_camera_int(entry, "fps", name, default_fps),
                }
            )
            spec["output_width"] = (
                int(spec["width"]) // 2
                if camera_type == "zedmini"
                else int(spec["width"])
            )
            spec["output_height"] = int(spec["height"])
        specs.append(spec)
    resolved_laptop_name = laptop_cam_name if laptop_camera else None
    if laptop_camera:
        for spec in specs:
            if spec["name"] == laptop_cam_name:
                spec["is_laptop"] = True
                spec["id"] = laptop_cam_id
                break
        else:
            specs.append(
                {"id": laptop_cam_id, "name": laptop_cam_name, "is_laptop": True}
            )
    return specs, resolved_laptop_name


def resolve_camera_ids(
    cam_ids: list[int | str] | None,
    rig_config: Path,
    *,
    camera_names: Sequence[str] | None = None,
) -> list[int | str]:
    names = list(camera_names or ("left_wrist", "right_wrist"))
    if cam_ids is not None:
        if camera_names is not None and len(cam_ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} camera IDs for {names}, got {len(cam_ids)}."
            )
        return cam_ids
    defaults = {"left_wrist": 0, "right_wrist": 2, "workspace": 4}
    data = load_rig_section(rig_config, "cameras")
    return [
        _read_camera_value(data, name, defaults.get(name, 0))
        for name in names
    ]


def connect_cameras(
    camera_specs: list[CameraSpec],
    *,
    fps: int,
    width: int,
    height: int,
    zero_non_laptop: bool,
    backend: str = "opencv",
) -> list[CameraDevice | None]:
    cameras: list[CameraDevice | None] = []
    for spec in camera_specs:
        cam_id = spec["id"]
        name = spec["name"]
        should_zero = zero_non_laptop and not spec["is_laptop"]
        if should_zero:
            cameras.append(None)
            log.info("Camera '%s' will be zero-filled.", name)
            continue

        camera_backend = str(spec.get("type", backend))
        camera_fps = int(spec.get("fps", fps))
        camera_width = int(spec.get("width", width))
        camera_height = int(spec.get("height", height))
        cam = make_camera_device(
            spec,
            default_backend=backend,
            default_fps=fps,
            default_width=width,
            default_height=height,
        )
        cam.connect()
        cameras.append(cam)
        label = " laptop overlay" if spec["is_laptop"] else ""
        log.info(
            "Camera '%s' (%s, index %s) connected at %dx%d/%d fps; output %dx%d.%s",
            name,
            camera_backend,
            cam_id,
            camera_width,
            camera_height,
            camera_fps,
            cam.output_width,
            cam.output_height,
            label,
        )
    return cameras


def read_camera_frames(
    cameras: list[CameraDevice | None],
    cam_names: list[str],
    *,
    width: int,
    height: int,
) -> dict:
    frames: dict = {}
    for cam, name in zip(cameras, cam_names):
        frame_width = int(getattr(cam, "output_width", width))
        frame_height = int(getattr(cam, "output_height", height))
        frame = (
            np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            if cam is None
            else cam.async_read()
        )
        frames[f"observation.images.{name}"] = frame
    return frames


def read_camera_samples(
    cameras: list[CameraDevice | None],
    cam_names: list[str],
    *,
    target_time_ns: int,
    record_time_ns: int,
    width: int,
    height: int,
    stale_timeout_s: float,
    max_sync_skew_s: float,
) -> tuple[dict, dict[str, bool]]:
    """Read camera frames nearest one clock target plus per-camera diagnostics."""
    frame: dict = {}
    health: dict[str, bool] = {}
    stale_timeout_ns = int(stale_timeout_s * 1e9)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)

    for camera, name in zip(cameras, cam_names):
        prefix = f"observation.camera.{name}"
        enabled = camera is not None
        frame_width = int(getattr(camera, "output_width", width))
        frame_height = int(getattr(camera, "output_height", height))
        try:
            sample = (
                CameraSample(
                    image=np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                    capture_time_ns=target_time_ns,
                    sequence=0,
                )
                if camera is None
                else camera.sample_at(target_time_ns)
            )
        except Exception as exc:
            log.debug("Camera '%s' read failed: %s", name, exc)
            sample = CameraSample(
                image=np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                capture_time_ns=0,
                sequence=0,
            )

        age_ns = (
            max(0, record_time_ns - sample.capture_time_ns)
            if sample.capture_time_ns
            else 2**63 - 1
        )
        sync_error_ns = (
            abs(sample.capture_time_ns - target_time_ns)
            if sample.capture_time_ns
            else 2**63 - 1
        )
        healthy = bool(
            not enabled
            or (
                sample.capture_time_ns > 0
                and age_ns <= stale_timeout_ns
                and sync_error_ns <= max_sync_skew_ns
            )
        )
        health[f"camera.{name}"] = healthy
        frame[f"observation.images.{name}"] = sample.image
        frame[f"{prefix}.healthy"] = _scalar_int(healthy if enabled else False)
        frame[f"{prefix}.sample_time_ns"] = _scalar_int(sample.capture_time_ns)
        frame[f"{prefix}.sequence"] = _scalar_int(sample.sequence)
    return frame, health


def disconnect_cameras(cameras: list[CameraDevice | None]) -> None:
    for cam in cameras:
        if cam is None:
            continue
        try:
            cam.disconnect()
        except Exception:
            pass


def make_camera_device(
    spec: CameraSpec,
    *,
    default_backend: str = "opencv",
    default_fps: int = 30,
    default_width: int = 640,
    default_height: int = 480,
) -> CameraDevice:
    """Construct one camera from a resolved spec without connecting it."""
    return _make_camera(
        str(spec.get("type", default_backend)),
        index_or_path=spec["id"],
        fps=int(spec.get("fps", default_fps)),
        width=int(spec.get("width", default_width)),
        height=int(spec.get("height", default_height)),
    )


def _make_camera(
    backend: str,
    *,
    index_or_path: int | str,
    fps: int,
    width: int,
    height: int,
) -> CameraDevice:
    normalized = backend.lower().replace("_", "-")
    if normalized in {"opencv", "cv2"}:
        return OpenCVCameraDevice(
            index_or_path=index_or_path,
            fps=fps,
            width=width,
            height=height,
        )
    if normalized in {"zedmini", "zed-mini"}:
        if width != 1344 or height != 376:
            raise ValueError(
                "ZED Mini requires the 1344x376 side-by-side UVC mode; "
                f"got {width}x{height}."
            )
        if fps not in {15, 30, 60, 100}:
            raise ValueError(
                "ZED Mini 1344x376 supports 15, 30, 60, or 100 fps; "
                f"got {fps}."
            )
        return ZedMiniCameraDevice(
            index_or_path=index_or_path,
            fps=fps,
            width=width,
            height=height,
        )
    raise ValueError(f"Unsupported camera backend {backend!r}.")


def camera_output_size(
    spec: CameraSpec,
    *,
    default_width: int,
    default_height: int,
) -> tuple[int, int]:
    """Return the image size exposed by one resolved camera specification."""
    return (
        int(spec.get("output_width", default_width)),
        int(spec.get("output_height", default_height)),
    )


def _normalize_camera_type(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"opencv", "cv2"}:
        return "opencv"
    if normalized in {"zedmini", "zed-mini"}:
        return "zedmini"
    raise SystemExit(
        f"Unsupported camera type {value!r}; expected 'opencv' or 'zedmini'."
    )


def _positive_camera_int(
    entry: dict[str, Any], key: str, name: str, default: int
) -> int:
    try:
        value = int(entry.get(key, default))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"cameras.{name}.{key} must be an integer.") from exc
    if value <= 0:
        raise SystemExit(f"cameras.{name}.{key} must be > 0.")
    return value


def _read_camera_value(data: dict[str, Any], key: str, default: int) -> int | str:
    section = data.get(key) or {}
    value = section.get("index_or_path", default)
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _scalar_int(value: int | bool) -> np.ndarray:
    return np.array([int(value)], dtype=np.int64)
