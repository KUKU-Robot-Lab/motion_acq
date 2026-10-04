"""quest-view: WebXR -> HandUMI conversion and the server chain to the tracking provider."""

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time

import numpy as np
import pytest

from motion_acq.head.retarget import hmd_yaw_pitch_deg
from motion_acq.quest_view.convert import handumi_frame
from motion_acq.tracking.transforms import unity_pose_to_handumi


def webxr_quat(yaw_left_deg: float = 0.0, pitch_up_deg: float = 0.0) -> list[float]:
    """WebXR orientation: yaw about +y (left), then pitch about +x (up); [x, y, z, w]."""
    y, p = math.radians(yaw_left_deg) / 2, math.radians(pitch_up_deg) / 2
    qy = np.array([0.0, math.sin(y), 0.0, math.cos(y)])
    qx = np.array([math.sin(p), 0.0, 0.0, math.cos(p)])
    x1, y1, z1, w1 = qy
    x2, y2, z2, w2 = qx
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def yaw_pitch_through_the_pipeline(frame: dict) -> tuple[float, float]:
    r, p = frame["hmdRotation"], frame["hmdPosition"]
    pose = unity_pose_to_handumi([p["x"], p["y"], p["z"]], [r["x"], r["y"], r["z"], r["w"]])
    yaw, pitch = hmd_yaw_pitch_deg(np.concatenate([pose.position, pose.quaternion]))
    return yaw, pitch


@pytest.mark.parametrize("yaw_left, pitch_up", [(25.0, 0.0), (-30.0, 0.0), (0.0, 20.0), (15.0, -10.0)])
def test_webxr_head_turns_reach_the_head_retargeter_with_the_same_sense(yaw_left, pitch_up):
    """Turning left / looking up in the headset must read as left / up where macq head reads it."""
    ref = handumi_frame({"hmd": {"p": [0, 1.5, 0], "q": webxr_quat()}})
    moved = handumi_frame({"hmd": {"p": [0, 1.5, 0], "q": webxr_quat(yaw_left, pitch_up)}})
    yaw0, pitch0 = yaw_pitch_through_the_pipeline(ref)
    yaw1, pitch1 = yaw_pitch_through_the_pipeline(moved)
    assert yaw1 - yaw0 == pytest.approx(yaw_left, abs=0.5)
    assert pitch1 - pitch0 == pytest.approx(pitch_up, abs=0.5)


def test_forward_and_position_axes():
    frame = handumi_frame({"hmd": {"p": [0.1, 1.6, -0.3], "q": webxr_quat()}})
    assert frame["hmdPosition"] == {"x": 0.1, "y": 1.6, "z": 0.3}  # 0.3 m ahead in Unity
    assert frame["hmdRotation"] == {"x": -0.0, "y": -0.0, "z": 0.0, "w": 1.0}


def test_controllers_buttons_and_missing_poses():
    message = {"hmd": None,
               "right": {"p": [0.2, 1.0, -0.4], "q": webxr_quat(), "buttons": [True, False, False, True, True, False],
                         "axes": [0, 0, 0.5, -1.0]},
               "left": {"p": [float("nan"), 0, 0], "q": webxr_quat()}}
    frame = handumi_frame(message)
    assert "hmdPosition" not in frame and "hmdRotation" not in frame  # untracked HMD, as the app
    assert frame["rightTracked"] and frame["rightTriggerPressed"] and frame["rightThumbstickClick"]
    assert frame["buttonAPressed"] and not frame["buttonBPressed"] and not frame["rightGripPressed"]
    assert frame["rightJoystick"] == {"x": 0.5, "y": 1.0, "z": 0.0}
    assert not frame["leftTracked"] and not frame["leftValid"]


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_turns_page_poses_into_the_handumi_stream():
    """Page -> websocket -> quest-view -> TCP HandUMI stream -> MetaQuestTrackingProvider."""
    import websockets.sync.client

    from motion_acq.quest_view.server import parse_args, serve
    from motion_acq.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider

    ports = {"port": free_port(), "tcp": free_port(), "sync": free_port(socket.SOCK_DGRAM),
             "head": free_port(socket.SOCK_DGRAM), "frames": free_port()}
    args = parse_args(["--port", str(ports["port"]), "--tcp-port", str(ports["tcp"]), "--sync-port",
                       str(ports["sync"]), "--head-status-port", str(ports["head"]), "--camera", "-1",
                       "--frame-port", str(ports["frames"])])
    loop = asyncio.new_event_loop()
    task = loop.create_task(serve(args))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    provider = None
    try:
        time.sleep(0.5)
        from motion_acq.calibration.control_tcp import ControllerTcpCalibration
        from motion_acq.robots.utils import IDENTITY_POSE7

        config = MetaQuestConfig(quest_ip="127.0.0.1", tcp_port=ports["tcp"], sync_port=ports["sync"])
        identity = np.asarray(IDENTITY_POSE7, dtype=np.float64)
        calibration = ControllerTcpCalibration(left=identity.copy(), right=identity.copy())
        provider = MetaQuestTrackingProvider(config=config, calibration=calibration, reset_workspace_on_x=False)
        provider.start()
        sample = provider.latest()
        with websockets.sync.client.connect(f"ws://127.0.0.1:{ports['port']}/ws") as ws:
            # head status arrives from the head process over UDP and is pushed to the page
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp.sendto(json.dumps({"locked": True, "state": "locked"}).encode(), ("127.0.0.1", ports["head"]))
            deadline = time.monotonic() + 5.0
            seen_status = False
            while time.monotonic() < deadline:
                ws.send(json.dumps({"type": "pose", "dt": 0.011, "hmd": {"p": [0, 1.5, 0], "q": webxr_quat(20, 0)},
                                    "right": {"p": [0.2, 1.1, -0.3], "q": webxr_quat()}}))
                try:
                    msg = ws.recv(timeout=0.05)
                    if isinstance(msg, str) and json.loads(msg).get("head", {}) and json.loads(msg)["head"].get("locked"):
                        seen_status = True
                except TimeoutError:
                    pass
                sample = provider.latest()
                if sample.streaming and sample.hmd_tracked and seen_status:
                    break
            assert seen_status, "head status never reached the page"
            assert sample.streaming and sample.hmd_tracked and sample.right_device_tracked
            yaw, _ = hmd_yaw_pitch_deg(np.asarray(sample.device_hmd_pose))
            assert yaw is not None
    finally:
        if provider is not None:
            provider.stop()
        async def shutdown():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run_coroutine_threadsafe(shutdown(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_frame_stream_reaches_the_recorder_camera():
    """quest-view test pattern -> TCP frame server -> QuestViewCameraDevice (RGB, monotonic time)."""
    import socket as _socket
    import subprocess
    import sys
    import time as _time

    from motion_acq.cameras.questview import QuestViewCameraDevice
    from motion_acq.cameras.usb import _make_camera

    def free_port() -> int:
        with _socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    page, frames = free_port(), free_port()
    with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        head = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "motion_acq.quest_view.server", "--port", str(page),
                             "--frame-port", str(frames), "--head-status-port", str(head),
                             "--test-pattern", "--no-pose-server", "--fps", "30"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        cam = _make_camera("quest-view", index_or_path=f"127.0.0.1:{frames}", fps=30, width=640, height=480)
        assert isinstance(cam, QuestViewCameraDevice)
        deadline = _time.monotonic() + 15
        while True:
            try:
                cam.connect()
                break
            except RuntimeError:
                if _time.monotonic() > deadline:
                    raise
                _time.sleep(0.3)
        before = _time.monotonic_ns()
        _time.sleep(0.3)
        sample = cam.sample_at(_time.monotonic_ns())
        assert sample.image.shape == (480, 640, 3)
        assert sample.capture_time_ns > before - 200_000_000
        # the pattern's box is BGR (0, 200, 255): in RGB the first channel is 255 and the last 0 there
        box = (sample.image[:, :, 0] == 255) & (sample.image[:, :, 1] == 200)
        assert box.any() and (sample.image[box][:, 2] == 0).all()
        cam.disconnect()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_frame_header_round_trip():
    from motion_acq.cameras.questview import parse_header
    from motion_acq.quest_view.server import FRAME_HEADER, FRAME_MAGIC

    raw = FRAME_HEADER.pack(FRAME_MAGIC, 7, 123456789, 640, 480, 3)
    assert parse_header(raw) == (7, 123456789, 640, 480, 3)
