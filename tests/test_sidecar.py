"""Sidecar streams (head/hand -> recorder): alignment, health, frames, features."""

from __future__ import annotations

import json
import math
import socket
import time

import numpy as np
import pytest

from motion_acq.hand.rh56f1 import load_rh56f1_map
from motion_acq.scripts.teleop_record import build_features
from motion_acq.sidecar import (
    HAND_JOINTS,
    SPECS,
    SidecarReceiver,
    hand_spec,
    head_spec,
    parse_sidecar_args,
    sidecar_features,
)

MS = 1_000_000


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def head_record(t: float, state: str = "running") -> dict:
    return {"t_mono_s": t, "state": state, "meas_pan_deg": 10.0, "meas_tilt_deg": 70.0,
            "cmd_pan_deg": 12.0, "cmd_tilt_deg": 71.0, "rel_yaw_deg": 5.0, "rel_pitch_deg": -3.0}


def test_hand_joint_order_matches_the_hand_map():
    assert HAND_JOINTS == load_rh56f1_map().joint_order


def test_nearest_record_within_skew_and_staleness():
    rx = SidecarReceiver(head_spec(), free_port())
    try:
        for k in range(10):
            rx.push(head_record(100.0 + k * 0.02))
        rec = rx.sample(int(100.081 * 1e9), 30 * MS)
        assert rec["t_mono_s"] == pytest.approx(100.08)
        assert rx.sample(int(100.5 * 1e9), 30 * MS) is None  # nothing near: stale
        frame, healthy = rx.frame(int(100.5 * 1e9), 30 * MS)
        assert not healthy and frame["observation.head.status"][0] == -1
        assert np.all(frame["observation.head.state"] == 0)
    finally:
        rx.close()


def test_out_of_order_and_malformed_records_are_rejected():
    rx = SidecarReceiver(head_spec(), free_port())
    try:
        rx.push(head_record(10.0))
        rx.push(head_record(9.0))
        rx.push({"state": "running"})
        rx.push({"t_mono_s": math.nan})
        assert rx.received == 1 and rx.rejected == 3
    finally:
        rx.close()


def test_head_frame_is_in_radians():
    spec = head_spec()
    frame = spec.to_frame(head_record(1.0))
    assert frame["observation.head.state"] == pytest.approx(np.deg2rad([10.0, 70.0]))
    assert frame["action.head"] == pytest.approx(np.deg2rad([12.0, 71.0]))
    assert frame["observation.head.hmd_rel"] == pytest.approx(np.deg2rad([5.0, -3.0]))
    assert frame["observation.head.status"].tolist() == [1]
    idle = spec.to_frame({"t_mono_s": 1.0, "state": "idle", "meas_pan_deg": 1.0, "meas_tilt_deg": 2.0,
                          "cmd_pan_deg": None, "rel_yaw_deg": None})
    assert idle["observation.head.status"].tolist() == [0]
    assert np.all(idle["action.head"] == 0)


def test_hand_frame_layout_and_status():
    spec = hand_spec("left")
    q = {j: 0.1 * i for i, j in enumerate(HAND_JOINTS)}
    record = {"t_mono_s": 1.0, "mode": "enabled", "state": "running", "q_command_rad": q,
              "measured_rad": {j: v + 0.01 for j, v in q.items()}, "glove_angles": list(range(20))}
    frame = spec.to_frame(record)
    assert frame["action.hand.left"] == pytest.approx([0.1 * i for i in range(6)])
    assert frame["observation.hand.left.state"] == pytest.approx([0.1 * i + 0.01 for i in range(6)])
    assert frame["observation.glove.left.angles"].tolist() == [float(i) for i in range(20)]
    assert frame["observation.hand.left.status"].tolist() == [1]
    assert spec.to_frame({**record, "mode": "fault"})["observation.hand.left.status"].tolist() == [3]
    assert spec.to_frame({**record, "mode": "disabled", "state": "idle"})["observation.hand.left.status"].tolist() == [0]
    broken = spec.to_frame({**record, "glove_angles": [1.0] * 19, "measured_rad": None})
    assert np.all(broken["observation.glove.left.angles"] == 0)
    assert np.all(broken["observation.hand.left.state"] == 0)


def test_features_match_frames():
    for name, make in SPECS.items():
        spec = make()
        frame = spec.to_frame(None)
        assert set(frame) == set(spec.features), name
        for key, feature in spec.features.items():
            assert frame[key].shape == tuple(feature["shape"]), key
            assert frame[key].dtype == np.dtype(feature["dtype"]), key
            assert len(feature["names"]) == feature["shape"][0], key
    left = hand_spec("left").features
    assert left["observation.hand.left.state"]["names"][0] == "l_hj_thumb_1"
    assert left["observation.glove.left.angles"]["names"][0] == "l_thumb_brake"


def test_build_features_includes_sidecars():
    features = build_features([], 640, 480, False, ["j1", "j2"], sidecar_names=["head", "hand_right"])
    assert "observation.head.state" in features and "action.hand.right" in features
    assert "observation.hand.left.state" not in features
    assert set(sidecar_features(["head"])) <= set(features)


def test_parse_sidecar_args():
    assert parse_sidecar_args(["head=47101", "hand_left=47112"]) == {"head": 47101, "hand_left": 47112}
    with pytest.raises(SystemExit, match="NAME=PORT"):
        parse_sidecar_args(["arm=1"])
    with pytest.raises(SystemExit, match="twice"):
        parse_sidecar_args(["head=1", "head=2"])


def test_udp_roundtrip():
    port = free_port()
    rx = SidecarReceiver(head_spec(), port)
    rx.start()
    try:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        now = time.monotonic()
        tx.sendto(json.dumps(head_record(now)).encode(), ("127.0.0.1", port))
        tx.sendto(b"not json", ("127.0.0.1", port))
        deadline = time.monotonic() + 2.0
        while rx.received < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        frame, ok = rx.frame(int(now * 1e9), 30 * MS)
        assert ok and frame["observation.head.status"].tolist() == [1]
        tx.close()
    finally:
        rx.close()


def test_keyboard_listener_owns_esc_space_r_q():
    import threading

    from motion_acq.teleop.common import KeyboardSpaceListener

    stop = threading.Event()
    keys = KeyboardSpaceListener(enabled=False, stop_event=stop)
    for char in (" ", "R", "q", "x"):
        keys.press(char)
    assert keys.consume_space() and not keys.consume_space()
    assert keys.consume_key("r") and keys.consume_key("q")
    assert not stop.is_set()
    keys.press("\x1b")
    assert stop.is_set()
