"""Head camera frames from macq quest-view (camera type "quest-view").

quest-view owns the head RealSense (the headset shows it), so the recorder
cannot open the device too. quest-view streams every raw frame on a local TCP
port (FRAME_HEADER + BGR pixels, capture time on this PC's monotonic clock);
this device keeps the last frames and serves the one nearest a requested time,
like the OpenCV backend.

rig:  cameras: {head: {type: quest-view, index_or_path: "127.0.0.1:47126", width: 640, height: 480, fps: 30}}
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from motion_acq.cameras.base import CameraDevice, CameraSample

FRAME_MAGIC = b"MQVF"
_HEADER_SIZE = 4 + 4 + 8 + 2 + 2 + 1  # quest_view.server.FRAME_HEADER ("<4sIQHHB")
CONNECT_TIMEOUT_S = 3.0
FIRST_FRAME_TIMEOUT_S = 3.0


def parse_header(raw: bytes) -> tuple[int, int, int, int, int]:
    """(sequence, capture_ns, width, height, channels) of one frame header."""
    import struct

    magic, seq, ns, width, height, channels = struct.unpack("<4sIQHHB", raw)
    if magic != FRAME_MAGIC:
        raise ValueError(f"not a quest-view frame (magic {magic!r})")
    return seq, ns, width, height, channels


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("quest-view closed the frame stream")
        buf += chunk
    return bytes(buf)


@dataclass
class QuestViewCameraDevice(CameraDevice):
    index_or_path: str  # "host:port"
    fps: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self._samples: deque[CameraSample] = deque(maxlen=16)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.error = ""

    @property
    def output_width(self) -> int:
        return int(self.width)

    @property
    def output_height(self) -> int:
        return int(self.height)

    def _address(self) -> tuple[str, int]:
        host, _, port = str(self.index_or_path).rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"quest-view camera address must be host:port, not {self.index_or_path!r}")
        return host, int(port)

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection(self._address(), timeout=CONNECT_TIMEOUT_S)
        except OSError as exc:
            raise RuntimeError(f"quest-view frame stream {self.index_or_path} not reachable ({exc}); "
                               "is macq quest-view running?") from exc
        self._sock.settimeout(2.0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="handumi_camera_quest_view", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                if self._samples:
                    sample = self._samples[-1]
                    break
            time.sleep(0.02)
        else:
            self.disconnect()
            raise RuntimeError(f"no frame from quest-view within {FIRST_FRAME_TIMEOUT_S} s (camera not open?)")
        h, w = sample.image.shape[:2]
        if (w, h) != (self.width, self.height):
            self.disconnect()
            raise RuntimeError(f"quest-view sends {w}x{h}, the rig expects {self.width}x{self.height}")

    def _run(self) -> None:
        assert self._sock is not None
        try:
            while not self._stop.is_set():
                seq, ns, width, height, channels = parse_header(_recv_exact(self._sock, _HEADER_SIZE))
                pixels = _recv_exact(self._sock, width * height * channels)
                bgr = np.frombuffer(pixels, np.uint8).reshape(height, width, channels)
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                with self._lock:
                    self._samples.append(CameraSample(image=rgb, capture_time_ns=int(ns), sequence=int(seq)))
        except (OSError, ValueError, ConnectionError) as exc:
            if not self._stop.is_set():
                self.error = str(exc)

    def async_read(self) -> np.ndarray:
        return self.sample_at().image

    def sample_at(self, target_time_ns: int | None = None) -> CameraSample:
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            raise RuntimeError(f"quest-view camera has no frame ({self.error or 'not connected'})")
        if target_time_ns is None:
            return samples[-1]
        return min(samples, key=lambda s: abs(s.capture_time_ns - int(target_time_ns)))

    def disconnect(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
