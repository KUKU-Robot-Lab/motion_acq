"""Sidecar streams: head and hand processes -> the arm recorder (one dataset).

The head (macq head) and hand (motion_acq_hand hand_node) processes send one
JSON datagram per control cycle to 127.0.0.1:<port>. Every record carries
``t_mono_s`` from time.monotonic(), the same CLOCK_MONOTONIC the recorder's
target_time_ns uses, so frames align by time across processes.

For each dataset frame the recorder asks every stream for the record nearest
the frame target time; a stream with no record within ``max_skew_s`` is
unhealthy for that frame (values zero, status -1) and feeds the recorder's
sensor health gate like a camera.

Feature layout (fixed shapes, LeRobot):
    head:        observation.head.state [pan, tilt] rad (measured)
                 action.head            [pan, tilt] rad (commanded)
                 observation.head.hmd_rel [yaw, pitch] rad (HMD relative to anchor)
                 observation.head.status [1] int64
    hand_<side>: observation.hand.<side>.state [6] rad (measured, RH56F1 joint order)
                 action.hand.<side>            [6] rad (commanded)
                 observation.glove.<side>.angles [20] rad (raw Nova 2 joints)
                 observation.hand.<side>.status [1] int64
status: -1 missing/stale, 0 idle/disabled, 1 running, 2 hold, 3 fault,
4 homing (hand walking to/from its home pose).
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import socket
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from motion_acq.hand.feedback import HAPTIC_JOINTS
from motion_acq.hand.nova2 import glove_joint_names

log = logging.getLogger(__name__)

# RH56F1 joint order of the hand map (configs/hands/rh56f1_hand_map.yaml joint_order).
HAND_JOINTS = ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1")
TIP_ORDER = ("thumb", "index", "middle", "ring", "pinky")
STATUS_MISSING = -1
_HEAD_STATUS = {"idle": 0, "locked": 0, "running": 1, "hold": 2}
_HAND_STATUS = {"idle": 0, "running": 1, "hold": 2, "homing": 4}


def _vector(values: Any, size: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


@dataclass(frozen=True)
class StreamSpec:
    name: str
    features: dict[str, dict[str, Any]]
    to_frame: Callable[[dict | None], dict[str, np.ndarray]]


def _feature(names: list[str], dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": (len(names),), "names": names}


def head_spec() -> StreamSpec:
    def to_frame(record: dict | None) -> dict[str, np.ndarray]:
        state = action = rel = None
        status = STATUS_MISSING
        if record is not None:
            state = _vector([record.get("meas_pan_deg"), record.get("meas_tilt_deg")], 2) \
                if record.get("meas_pan_deg") is not None else None
            action = _vector([record.get("cmd_pan_deg"), record.get("cmd_tilt_deg")], 2) \
                if record.get("cmd_pan_deg") is not None else None
            rel = _vector([record.get("rel_yaw_deg"), record.get("rel_pitch_deg")], 2) \
                if record.get("rel_yaw_deg") is not None else None
            status = _HEAD_STATUS.get(str(record.get("state")), STATUS_MISSING)
        zeros = np.zeros(2, dtype=np.float32)
        return {
            "observation.head.state": np.deg2rad(state).astype(np.float32) if state is not None else zeros,
            "action.head": np.deg2rad(action).astype(np.float32) if action is not None else zeros,
            "observation.head.hmd_rel": np.deg2rad(rel).astype(np.float32) if rel is not None else zeros,
            "observation.head.status": np.array([status], dtype=np.int64),
        }

    return StreamSpec("head", {
        "observation.head.state": _feature(["head_pan", "head_tilt"]),
        "action.head": _feature(["head_pan", "head_tilt"]),
        "observation.head.hmd_rel": _feature(["hmd_rel_yaw", "hmd_rel_pitch"]),
        "observation.head.status": _feature(["status"], "int64"),
    }, to_frame)


def hand_spec(side: str) -> StreamSpec:
    prefix = {"right": "r", "left": "l"}[side]
    joints = [f"{prefix}_hj_{j}" for j in HAND_JOINTS]
    glove = list(glove_joint_names(side))

    def to_frame(record: dict | None) -> dict[str, np.ndarray]:
        state = action = angles = tips = jforce = haptics = None
        status = STATUS_MISSING
        if record is not None:
            tip = record.get("tip_force_n")
            tips = _vector([tip.get(f) for f in TIP_ORDER], 5) if tip else None
            force = record.get("joint_force")
            jforce = _vector([force.get(j) for j in HAND_JOINTS], 6) if force else None
            hap = record.get("haptics")
            if hap:
                levels = ([hap["brake"].get(b) for b in ("thumb", "index", "middle", "ring")] + [hap.get("squeeze")]
                          + [hap["vibration"].get(v) for v in HAPTIC_JOINTS[5:]])
                haptics = _vector(levels, len(HAPTIC_JOINTS))
            measured = record.get("measured_rad")
            state = _vector([measured[j] for j in HAND_JOINTS], 6) if measured else None
            command = record.get("q_command_rad")
            action = _vector([command[j] for j in HAND_JOINTS], 6) if command else None
            angles = _vector(record.get("glove_angles"), 20)
            if record.get("mode") == "fault":
                status = 3
            elif record.get("mode") == "disabled":
                status = 0
            else:
                status = _HAND_STATUS.get(str(record.get("state")), STATUS_MISSING)
        return {
            f"observation.hand.{side}.state": state if state is not None else np.zeros(6, np.float32),
            f"action.hand.{side}": action if action is not None else np.zeros(6, np.float32),
            f"observation.glove.{side}.angles": angles if angles is not None else np.zeros(20, np.float32),
            f"observation.hand.{side}.status": np.array([status], dtype=np.int64),
            f"observation.hand.{side}.tip_force": tips if tips is not None else np.zeros(5, np.float32),
            f"observation.hand.{side}.joint_force": jforce if jforce is not None else np.zeros(6, np.float32),
            f"action.glove.{side}.haptics": haptics if haptics is not None else np.zeros(len(HAPTIC_JOINTS), np.float32),
        }

    return StreamSpec(f"hand_{side}", {
        f"observation.hand.{side}.state": _feature(joints),
        f"action.hand.{side}": _feature(joints),
        f"observation.glove.{side}.angles": _feature(glove),
        f"observation.hand.{side}.status": _feature(["status"], "int64"),
        # RH56F1 sensors (10.05): tip normal force (N), motor force per joint (driver units, g)
        f"observation.hand.{side}.tip_force": _feature([f"{prefix}_tip_{f}" for f in TIP_ORDER]),
        f"observation.hand.{side}.joint_force": _feature([f"{prefix}_hf_{j}" for j in HAND_JOINTS]),
        # Nova 2 feedback sent (0..1): brakes, strap, vibration (motion_acq.hand.feedback.HAPTIC_JOINTS)
        f"action.glove.{side}.haptics": _feature([f"{prefix}_{j}" for j in HAPTIC_JOINTS]),
    }, to_frame)


SPECS: dict[str, Callable[[], StreamSpec]] = {
    "head": head_spec,
    "hand_right": lambda: hand_spec("right"),
    "hand_left": lambda: hand_spec("left"),
}


class SidecarReceiver:
    """UDP JSON receiver keeping a short time-indexed history of one stream."""

    def __init__(self, spec: StreamSpec, port: int, *, history: int = 512,
                 host: str = "127.0.0.1") -> None:
        self.spec = spec
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, self.port))
        self._sock.settimeout(0.2)
        self._times: deque[int] = deque(maxlen=history)
        self._records: deque[dict] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.received = 0
        self.rejected = 0
        self._thread = threading.Thread(target=self._run, name=f"sidecar-{spec.name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._sock.close()

    def push(self, record: dict) -> None:
        t = record.get("t_mono_s")
        if not isinstance(t, (int, float)) or not math.isfinite(t):
            self.rejected += 1
            return
        t_ns = int(t * 1e9)
        with self._lock:
            if self._times and t_ns < self._times[-1]:
                self.rejected += 1  # out of order: keep the history sorted
                return
            self._times.append(t_ns)
            self._records.append(record)
            self.received += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._sock.recv(65535)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                break
            try:
                record = json.loads(data)
            except ValueError:
                self.rejected += 1
                continue
            if isinstance(record, dict):
                self.push(record)

    def sample(self, target_time_ns: int, max_skew_ns: int) -> dict | None:
        with self._lock:
            times = list(self._times)
            if not times:
                return None
            i = bisect.bisect_left(times, target_time_ns)
            candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
            best = min(candidates, key=lambda j: abs(times[j] - target_time_ns))
            if abs(times[best] - target_time_ns) > max_skew_ns:
                return None
            return self._records[best]

    def frame(self, target_time_ns: int, max_skew_ns: int) -> tuple[dict[str, np.ndarray], bool]:
        record = self.sample(target_time_ns, max_skew_ns)
        return self.spec.to_frame(record), record is not None


def parse_udp_targets(text: str) -> list[tuple[str, int]]:
    """'127.0.0.1:47111,127.0.0.1:47141' -> [(host, port), ...]; '' -> []."""
    targets = []
    for item in (part.strip() for part in str(text or "").split(",")):
        if not item:
            continue
        host, sep, port = item.rpartition(":")
        if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError(f"udp target must be HOST:PORT, not {item!r}")
        targets.append((host, int(port)))
    return targets


def parse_sidecar_args(values: list[str] | None) -> dict[str, int]:
    """``--sidecar head=47101 --sidecar hand_right=47111`` -> {name: port}."""
    out: dict[str, int] = {}
    for value in values or []:
        name, sep, port = value.partition("=")
        if not sep or name not in SPECS:
            raise SystemExit(f"--sidecar expects NAME=PORT with NAME in {sorted(SPECS)}: {value!r}")
        if name in out:
            raise SystemExit(f"--sidecar {name} given twice")
        out[name] = int(port)
    return out


def sidecar_features(names: list[str]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for name in names:
        features.update(SPECS[name]().features)
    return features
