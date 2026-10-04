"""macq quest-view: the head camera in the Quest headset, its poses back to the PC.

    MACQ_STATION=arm4090 macq quest-view            # then: scripts/quest_usb.sh view

Replaces the HandUMI Quest App while the operator needs to see the head
camera. A WebXR page in the Quest Browser (served here, reached over USB by
adb reverse as http://localhost:<port>, a secure context without
certificates) shows the head RealSense image head-locked in front of the
eyes with a status line, and sends the HMD and controller poses back every
XR frame. This server re-publishes them in the HandUMI wire format on
TCP 127.0.0.1:65432 (and answers its UDP time-sync), so macq head,
teleop-real and teleop-record connect to it unchanged (quest_ip 127.0.0.1).
Remove the HandUMI adb forward first: both use local port 65432.

The head process sends its records here with --udp-target 127.0.0.1:47121;
the status line shows locked / following, the HMD offset from the anchor
and the camera pan/tilt from home against the window.

The head camera is opened only while it is needed (user 10.04: no RealSense when
the headset connects, the picture from the moment the neck starts): while head
records arrive on UDP 47121 (macq head runs) or a recorder reads frames
(--camera-on always keeps it open). quest-view owns the camera, so it also
hands every frame to the recorder:
TCP 127.0.0.1:47126 streams raw BGR frames (FRAME_HEADER + pixels) to each
connected client (motion_acq.cameras.questview, camera type "quest-view").
--test-pattern replaces the camera with a moving pattern and --no-pose-server
leaves 65432 to the mock sender (fake recording with video).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import WSMsgType, web

from motion_acq.quest_view.convert import handumi_frame

log = logging.getLogger("motion_acq.quest_view")

WEB_DIR = Path(__file__).resolve().parent / "web"
_PING = struct.Struct("<BQ")  # HandUMI time-sync (motion_acq.tracking.meta_quest)
_PONG = struct.Struct("<BQQ")
HEAD_STATUS_PORT = 47121
FRAME_PORT = 47126
# magic, sequence, capture time (time.monotonic_ns), width, height, channels; then height*width*channels bytes (BGR)
FRAME_HEADER = struct.Struct("<4sIQHHB")
FRAME_MAGIC = b"MQVF"
STATUS_PERIOD_S = 0.1
NO_POSE_WARN_S = 10.0
HEAD_FRESH_S = 2.0  # head records younger than this keep the camera open


@dataclass
class Shared:
    """State shared between the camera thread and the asyncio side."""

    jpeg: bytes | None = None
    jpeg_seq: int = 0
    frame: bytes | None = None  # raw BGR of the same frame, for the recorder
    frame_shape: tuple[int, int, int] = (0, 0, 0)
    frame_ns: int = 0
    camera_on: bool = False
    head: dict | None = None
    frames_in: int = 0
    last_pose_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _publish(shared: Shared, frame, params, captured_ns: int) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, params)
    raw = frame.tobytes()
    with shared.lock:
        if ok:
            shared.jpeg = buf.tobytes()
        shared.frame, shared.frame_shape, shared.frame_ns = raw, tuple(frame.shape), captured_ns
        shared.jpeg_seq += 1


def _camera_off(shared: Shared) -> None:
    with shared.lock:
        shared.camera_on = False
        shared.jpeg, shared.frame = None, None


def test_pattern_thread(width: int, height: int, fps: float, quality: int, shared: Shared,
                        stop: threading.Event, want=lambda: True) -> None:
    """A moving pattern instead of the camera (fake runs: recording with video, no hardware)."""
    import cv2
    import numpy as np

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    period = 1.0 / max(fps, 1.0)
    n = 0
    while not stop.is_set():
        if not want():
            if shared.camera_on:
                _camera_off(shared)
            stop.wait(0.1)
            continue
        shared.camera_on = True
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 1] = 40
        x = int((n * 7) % width)
        cv2.rectangle(frame, (x, height // 3), (min(x + 60, width - 1), 2 * height // 3), (0, 200, 255), -1)
        cv2.putText(frame, f"quest-view test {n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        _publish(shared, frame, params, time.monotonic_ns())
        n += 1
        stop.wait(period)


def camera_thread(index: int, width: int, height: int, fps: float, quality: int,
                  shared: Shared, stop: threading.Event, want=lambda: True) -> None:
    """Open the camera while want() (the neck runs or a recorder reads), release it otherwise."""
    import cv2

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cap = None
    try:
        while not stop.is_set():
            if not want():
                if cap is not None:
                    cap.release()
                    cap = None
                    _camera_off(shared)
                    log.info("camera %d closed (neck stopped, no recorder)", index)
                stop.wait(0.2)
                continue
            if cap is None:
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                if not cap.isOpened():
                    log.error("camera %d did not open: no video in the headset; retrying", index)
                    cap.release()
                    cap = None
                    stop.wait(2.0)
                    continue
                shared.camera_on = True
                log.info("camera %d open: %dx%d @ %g fps", index, width, height, fps)
            ok, frame = cap.read()
            captured_ns = time.monotonic_ns()
            if not ok:
                time.sleep(0.05)
                continue
            _publish(shared, frame, params, captured_ns)
    finally:
        if cap is not None:
            cap.release()


class FrameServer:
    """Raw frames to local readers (the recorder). A slow reader skips frames, never blocks the camera."""

    def __init__(self, shared: Shared) -> None:
        self.shared = shared
        self.clients = 0

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients += 1
        log.info("frame reader connected (recorder)")
        sent = 0
        try:
            while not writer.is_closing():
                with self.shared.lock:
                    seq, raw, shape, ns = (self.shared.jpeg_seq, self.shared.frame, self.shared.frame_shape,
                                           self.shared.frame_ns)
                if raw is None or seq == sent:
                    await asyncio.sleep(0.005)
                    continue
                h, w, c = shape
                writer.write(FRAME_HEADER.pack(FRAME_MAGIC, seq & 0xFFFFFFFF, ns, w, h, c) + raw)
                await writer.drain()
                sent = seq
        except (ConnectionError, OSError):
            pass
        finally:
            self.clients -= 1
            writer.close()
            log.info("frame reader left")


class PoseBroadcast:
    """TCP server in the HandUMI wire format (JSON lines) for the tracking providers."""

    def __init__(self) -> None:
        self.writers: set[asyncio.StreamWriter] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        log.info("tracking client connected: %s", peer)
        self.writers.add(writer)
        try:
            await reader.read()  # clients only listen; returns on close
        finally:
            self.writers.discard(writer)
            writer.close()
            log.info("tracking client left: %s", peer)

    def send(self, frame: dict) -> None:
        line = (json.dumps(frame) + "\n").encode("utf-8")
        for writer in list(self.writers):
            if writer.is_closing():
                self.writers.discard(writer)
                continue
            if writer.transport.get_write_buffer_size() > 1 << 20:
                continue  # a stalled client must not grow memory; it gets the next frame
            writer.write(line)


class SyncResponder(asyncio.DatagramProtocol):
    """Answers the providers' clock pings with this PC's monotonic clock (one clock)."""

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) == _PING.size:
            kind, t1 = _PING.unpack(data)
            if kind == 1:
                self.transport.sendto(_PONG.pack(2, t1, time.monotonic_ns()), addr)


class HeadStatus(asyncio.DatagramProtocol):
    def __init__(self, shared: Shared) -> None:
        self.shared = shared

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            record = json.loads(data)
        except ValueError:
            return
        if isinstance(record, dict):
            record["received_at"] = time.monotonic()
            self.shared.head = record


def build_app(shared: Shared, poses: PoseBroadcast, *, video_fps: float) -> web.Application:
    app = web.Application()

    async def index(_request):
        # The Quest Browser caches scripts hard; version the script by its mtime and
        # never cache the page, so an updated viewer.js is always loaded.
        version = int((WEB_DIR / "viewer.js").stat().st_mtime)
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("/static/viewer.js", f"/static/viewer.js?v={version}")
        return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=1 << 20)
        await ws.prepare(request)
        log.info("headset page connected")
        sender = asyncio.create_task(_push(ws, shared, video_fps))
        try:
            async for msg in ws:
                if msg.type is WSMsgType.TEXT:
                    try:
                        message = json.loads(msg.data)
                    except ValueError:
                        continue
                    if isinstance(message, dict) and message.get("type") == "pose":
                        poses.send(handumi_frame(message))
                        shared.frames_in += 1
                        shared.last_pose_at = time.monotonic()
                elif msg.type is WSMsgType.ERROR:
                    break
        finally:
            sender.cancel()
            log.info("headset page left")
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", WEB_DIR)
    return app


async def _push(ws: web.WebSocketResponse, shared: Shared, video_fps: float) -> None:
    """Latest camera frame (binary) and head status (text) to one page."""
    sent_seq = 0
    next_status = 0.0
    period = 1.0 / max(video_fps, 1.0)
    while not ws.closed:
        with shared.lock:
            jpeg, seq = shared.jpeg, shared.jpeg_seq
        if jpeg is not None and seq != sent_seq:
            await ws.send_bytes(jpeg)
            sent_seq = seq
        now = time.monotonic()
        if now >= next_status:
            next_status = now + STATUS_PERIOD_S
            head = shared.head
            age = None if head is None else now - head.get("received_at", 0.0)
            await ws.send_str(json.dumps({"type": "status", "head": head, "head_age_s": age,
                                          "camera_on": shared.camera_on}))
        await asyncio.sleep(period)


async def serve(args: argparse.Namespace) -> None:
    shared = Shared()
    stop = threading.Event()
    frames = FrameServer(shared)

    def want_camera() -> bool:
        if args.camera_on == "always" or frames.clients > 0:
            return True
        head = shared.head
        return head is not None and time.monotonic() - float(head.get("received_at", 0.0)) < HEAD_FRESH_S

    if args.test_pattern:
        threading.Thread(target=test_pattern_thread, daemon=True, name="quest-view-pattern",
                         args=(args.width, args.height, args.fps, args.jpeg_quality, shared, stop, want_camera)
                         ).start()
    elif args.camera >= 0:
        threading.Thread(target=camera_thread, daemon=True, name="quest-view-camera",
                         args=(args.camera, args.width, args.height, args.fps, args.jpeg_quality, shared, stop,
                               want_camera)).start()
    poses = PoseBroadcast()
    loop = asyncio.get_running_loop()
    tcp = None
    if not args.no_pose_server:
        try:
            tcp = await asyncio.start_server(poses.handle, "127.0.0.1", args.tcp_port)
        except OSError as exc:
            raise SystemExit(f"TCP 127.0.0.1:{args.tcp_port} is taken ({exc}); remove the HandUMI forward: "
                             f"adb forward --remove tcp:{args.tcp_port}") from exc
        await loop.create_datagram_endpoint(SyncResponder, local_addr=("127.0.0.1", args.sync_port))
    frame_tcp = await asyncio.start_server(frames.handle, "127.0.0.1", args.frame_port)
    await loop.create_datagram_endpoint(lambda: HeadStatus(shared), local_addr=("127.0.0.1", args.head_status_port))
    runner = web.AppRunner(build_app(shared, poses, video_fps=args.fps), shutdown_timeout=1.0)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    log.info("page http://localhost:%d (Quest: scripts/quest_usb.sh view); poses -> TCP 127.0.0.1:%s; "
             "head status <- UDP %d; frames -> TCP 127.0.0.1:%d", args.port,
             "off" if tcp is None else args.tcp_port, args.head_status_port, args.frame_port)
    started = time.monotonic()
    try:
        while True:
            await asyncio.sleep(5.0)
            now = time.monotonic()
            age = now - shared.last_pose_at if shared.last_pose_at else None
            log.info("poses in %d (last %s ago), tracking clients %d, camera frames %d, frame readers %d",
                     shared.frames_in, f"{age:.1f} s" if age is not None else "never",
                     len(poses.writers), shared.jpeg_seq, frames.clients)
            if tcp is not None and (age if age is not None else now - started) > NO_POSE_WARN_S:
                log.warning("no headset poses for %.0f s: VR not started, or a USB re-plug dropped the adb "
                            "reverse; run scripts/quest_usb.sh view (or vr) again",
                            age if age is not None else now - started)
    finally:
        stop.set()
        if tcp is not None:
            tcp.close()
        frame_tcp.close()
        await runner.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8787, help="page and websocket (adb reverse)")
    p.add_argument("--tcp-port", type=int, default=65432, help="HandUMI pose stream for the providers")
    p.add_argument("--sync-port", type=int, default=42000)
    p.add_argument("--head-status-port", type=int, default=HEAD_STATUS_PORT)
    p.add_argument("--camera", type=int, default=4, help="V4L2 index of the head camera, -1 = none")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--frame-port", type=int, default=FRAME_PORT, help="raw frames for the recorder")
    p.add_argument("--test-pattern", action="store_true", help="moving pattern instead of the camera (fake)")
    p.add_argument("--camera-on", choices=("demand", "always"), default="demand",
                   help="demand: only while the neck runs or a recorder reads (default)")
    p.add_argument("--no-pose-server", action="store_true",
                   help="no TCP 65432 / sync (the mock sender serves poses; fake recording with video)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    from motion_acq.cpu import keep_off_rt

    log.info(keep_off_rt())
    try:
        asyncio.run(serve(parse_args(argv)))
    except KeyboardInterrupt:
        log.info("quest-view stopped")


if __name__ == "__main__":
    main()
