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
STATUS_PERIOD_S = 0.1


@dataclass
class Shared:
    """State shared between the camera thread and the asyncio side."""

    jpeg: bytes | None = None
    jpeg_seq: int = 0
    head: dict | None = None
    frames_in: int = 0
    last_pose_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def camera_thread(index: int, width: int, height: int, fps: float, quality: int,
                  shared: Shared, stop: threading.Event) -> None:
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        log.error("camera %d did not open: no video in the headset", index)
        return
    log.info("camera %d: %dx%d @ %g fps", index, width, height, fps)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, params)
            if ok:
                with shared.lock:
                    shared.jpeg, shared.jpeg_seq = buf.tobytes(), shared.jpeg_seq + 1
    finally:
        cap.release()


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
        return web.FileResponse(WEB_DIR / "index.html")

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
            await ws.send_str(json.dumps({"type": "status", "head": head, "head_age_s": age}))
        await asyncio.sleep(period)


async def serve(args: argparse.Namespace) -> None:
    shared = Shared()
    stop = threading.Event()
    if args.camera >= 0:
        threading.Thread(target=camera_thread, daemon=True, name="quest-view-camera",
                         args=(args.camera, args.width, args.height, args.fps, args.jpeg_quality, shared, stop)
                         ).start()
    poses = PoseBroadcast()
    loop = asyncio.get_running_loop()
    try:
        tcp = await asyncio.start_server(poses.handle, "127.0.0.1", args.tcp_port)
    except OSError as exc:
        raise SystemExit(f"TCP 127.0.0.1:{args.tcp_port} is taken ({exc}); remove the HandUMI forward: "
                         f"adb forward --remove tcp:{args.tcp_port}") from exc
    await loop.create_datagram_endpoint(SyncResponder, local_addr=("127.0.0.1", args.sync_port))
    await loop.create_datagram_endpoint(lambda: HeadStatus(shared), local_addr=("127.0.0.1", args.head_status_port))
    runner = web.AppRunner(build_app(shared, poses, video_fps=args.fps))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    log.info("page http://localhost:%d (Quest: scripts/quest_usb.sh view); poses -> TCP 127.0.0.1:%d; "
             "head status <- UDP %d", args.port, args.tcp_port, args.head_status_port)
    try:
        while True:
            await asyncio.sleep(5.0)
            age = time.monotonic() - shared.last_pose_at if shared.last_pose_at else None
            log.info("poses in %d (last %s ago), tracking clients %d, camera frames %d",
                     shared.frames_in, f"{age:.1f} s" if age is not None else "never",
                     len(poses.writers), shared.jpeg_seq)
    finally:
        stop.set()
        tcp.close()
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
