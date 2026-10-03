"""ZED Mini UVC camera backend exposing only its left image."""

from __future__ import annotations

import numpy as np

from motion_acq.cameras.opencv import OpenCVCameraDevice


class ZedMiniCameraDevice(OpenCVCameraDevice):
    """Capture a side-by-side ZED Mini frame and retain the left half.

    The supported UVC mode is 1344x376. Its two 672x376 eye images are packed
    horizontally, so downstream recording and visualization receive only
    ``image[:, :672]``. ``index_or_path`` may be an OpenCV integer index such
    as ``0`` or an explicit device path such as ``/dev/video0``.
    """

    @property
    def output_width(self) -> int:
        return int(self.width) // 2

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim < 2:
            raise ValueError(f"Invalid ZED Mini frame shape: {frame.shape}")
        expected_shape = (int(self.height), int(self.width))
        if tuple(frame.shape[:2]) != expected_shape:
            raise ValueError(
                "ZED Mini did not honor the configured capture mode: "
                f"expected {expected_shape[1]}x{expected_shape[0]}, "
                f"got {frame.shape[1]}x{frame.shape[0]}."
            )
        actual_width = int(frame.shape[1])
        if actual_width < 2 or actual_width % 2:
            raise ValueError(
                f"ZED Mini side-by-side frame width must be even, got {actual_width}."
            )
        return frame[:, : actual_width // 2].copy()


__all__ = ["ZedMiniCameraDevice"]
