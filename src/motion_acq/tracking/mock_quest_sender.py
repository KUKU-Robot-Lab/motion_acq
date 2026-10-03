"""Mock Quest app for developing the Phase 2A receiver without hardware.

Emulates the native Quest app end of the contract so the whole pipe — TCP/JSON
pose stream + UDP NTP-style time-sync — can be exercised on the workstation:

  * TCP server: accepts a connection and streams newline-delimited JSON pose
    samples (raw Unity coordinates) at a fixed rate. Controllers gently
    oscillate so the receiver shows changing numbers.
  * UDP server: echoes time-sync pings with a *device clock* that is offset from
    the PC clock by ``--skew-s`` seconds, so the receiver's offset estimate is
    non-zero and verifiable.

Run this in one terminal, then the receiver in another:

    PYTHONPATH=src python -m motion_acq.tracking.mock_quest_sender
    PYTHONPATH=src python -m motion_acq.tracking.meta_quest

See docs/phase-2-motion-tracking.md → TCP/JSON Payload for the field layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass

log = logging.getLogger("motion_acq.tracking.mock_quest_sender")

_PING = struct.Struct("<BQ")  # (msg_type=1, t1_pc_ns)
_PONG = struct.Struct("<BQQ")  # (msg_type=2, t1_echo, t2_quest_ns)
_PING_TYPE = 1
_PONG_TYPE = 2


def _device_time_ns(skew_ns: int) -> int:
    """Fake Quest monotonic clock = PC monotonic clock + a fixed skew."""
    return time.monotonic_ns() + skew_ns


def _xyz(x: float, y: float, z: float) -> dict:
    return {"x": x, "y": y, "z": z}


def _xyzw(x: float, y: float, z: float, w: float) -> dict:
    return {"x": x, "y": y, "z": z, "w": w}


@dataclass(frozen=True)
class HmdMotion:
    """Optional head motion for exercising the head pipeline without a Quest.

    Yaw turns about Unity +y (positive = look right), pitch about Unity +x
    (positive = look down). Every ``loss_every_s`` the HMD pose is dropped for
    ``loss_s`` to exercise tracking-loss HOLD.
    """

    yaw_amp_deg: float = 0.0
    pitch_amp_deg: float = 0.0
    period_s: float = 8.0
    loss_every_s: float = 0.0
    loss_s: float = 0.0

    def rotation(self, t: float) -> dict:
        phase = 2.0 * math.pi * t / self.period_s if self.period_s > 0 else 0.0
        yaw = math.radians(self.yaw_amp_deg * math.sin(phase))
        pitch = math.radians(self.pitch_amp_deg * math.sin(2.0 * phase))
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        # q_yaw(about y) * q_pitch(about x), quaternions as (x, y, z, w).
        return _xyzw(cy * sp, sy * cp, -sy * sp, cy * cp)

    def lost(self, t: float) -> bool:
        if self.loss_every_s <= 0.0 or self.loss_s <= 0.0:
            return False
        return (t % self.loss_every_s) >= self.loss_every_s - self.loss_s


def _make_frame(
    seq: int, t0: float, skew_ns: int, hmd: HmdMotion | None = None
) -> dict:
    """Build one HandUMI Quest App wire sample in raw Unity coordinates."""
    hmd = hmd or HmdMotion()
    t = time.monotonic() - t0
    sway = 0.05 * math.sin(t)
    bob = 0.05 * math.sin(2.0 * t)
    reach = 0.05 * math.cos(t)
    frame = {
        # Top-level timing (the compatibility TCP/JSON format has no sequence).
        "ovrTimeNs": _device_time_ns(skew_ns),
        "deltaTime": 1.0 / 72.0,
        # HMD pose.
        "hmdPosition": _xyz(0.0, 1.10, 0.05),
        "hmdRotation": hmd.rotation(t),
        # Left controller.
        "leftControllerPosition": _xyz(-0.20 + sway, 0.95 + bob, 0.30 + reach),
        "leftControllerRotation": _xyzw(0.0, 0.0, 0.0, 1.0),
        "leftTracked": True,
        "leftValid": True,
        "leftJoystick": _xyz(0.0, 0.0, 0.0),
        "leftThumbstickClick": False,
        "leftTriggerPressed": False,
        "leftGripPressed": False,
        "buttonXPressed": False,
        "buttonYPressed": False,
        # Right controller.
        "rightControllerPosition": _xyz(0.20 - sway, 0.95 + bob, 0.30 - reach),
        "rightControllerRotation": _xyzw(0.0, 0.0, 0.0, 1.0),
        "rightTracked": True,
        "rightValid": True,
        "rightJoystick": _xyz(0.0, 0.0, 0.0),
        "rightThumbstickClick": False,
        "rightTriggerPressed": False,
        "rightGripPressed": False,
        "buttonAPressed": False,
        "buttonBPressed": False,
        # Battery.
        "hmdBattPct": 87,
        "leftBattPct": 90,
        "rightBattPct": 92,
        "hmdCharging": False,
    }
    if hmd.lost(t):
        # The receiver reports the HMD untracked when its pose keys are absent.
        del frame["hmdPosition"], frame["hmdRotation"]
    return frame


def _udp_sync_server(host: str, sync_port: int, skew_ns: int, stop: threading.Event) -> None:
    """Echo every ping with the device clock (the Quest end of time-sync)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, sync_port))
    sock.settimeout(0.5)
    log.info("UDP time-sync server on %s:%d", host, sync_port)
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(32)
            except TimeoutError:
                continue
            except OSError:
                break
            if len(data) != _PING.size:
                continue
            msg_type, t1 = _PING.unpack(data)
            if msg_type != _PING_TYPE:
                continue
            sock.sendto(_PONG.pack(_PONG_TYPE, t1, _device_time_ns(skew_ns)), addr)
    finally:
        sock.close()


def _serve_client(conn: socket.socket, addr, fps: float, skew_ns: int,
                  stop: threading.Event, hmd: HmdMotion | None = None,
                  t0: float | None = None) -> None:
    log.info("Client connected: %s", addr)
    seq = 0
    t0 = time.monotonic() if t0 is None else t0  # shared clock: all clients see one motion
    period = 1.0 / fps if fps > 0 else 0.0
    conn.settimeout(1.0)
    try:
        while not stop.is_set():
            loop_start = time.monotonic()
            frame = _make_frame(seq, t0, skew_ns, hmd)
            line = (json.dumps(frame) + "\n").encode("utf-8")
            try:
                conn.sendall(line)
            except OSError:
                break
            seq += 1
            dt = time.monotonic() - loop_start
            time.sleep(max(period - dt, 0.0))
    finally:
        conn.close()
        log.info("Client disconnected: %s", addr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Quest app (TCP/JSON + UDP sync).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=65432)
    parser.add_argument("--sync-port", type=int, default=42000)
    parser.add_argument("--fps", type=float, default=72.0)
    parser.add_argument("--skew-s", type=float, default=5.0,
                        help="Fake device-clock skew vs PC clock (verifies sync).")
    parser.add_argument("--hmd-yaw-amp-deg", type=float, default=0.0)
    parser.add_argument("--hmd-pitch-amp-deg", type=float, default=0.0)
    parser.add_argument("--hmd-period-s", type=float, default=8.0)
    parser.add_argument("--hmd-loss-every-s", type=float, default=0.0,
                        help="Drop the HMD pose periodically (0 = never).")
    parser.add_argument("--hmd-loss-s", type=float, default=0.0)
    args = parser.parse_args()
    hmd = HmdMotion(
        yaw_amp_deg=args.hmd_yaw_amp_deg,
        pitch_amp_deg=args.hmd_pitch_amp_deg,
        period_s=args.hmd_period_s,
        loss_every_s=args.hmd_loss_every_s,
        loss_s=args.hmd_loss_s,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    skew_ns = int(args.skew_s * 1e9)
    stop = threading.Event()

    udp_thread = threading.Thread(
        target=_udp_sync_server,
        args=(args.host, args.sync_port, skew_ns, stop),
        daemon=True,
    )
    udp_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.tcp_port))
    # Like the HandUMI Quest App (PoseSender keeps a client list), serve every
    # client at once: the arm recorder and the head process connect together.
    server.listen(4)
    server.settimeout(0.5)
    log.info("TCP pose server on %s:%d (fps=%.0f, skew=%.1fs). Ctrl+C to stop.",
             args.host, args.tcp_port, args.fps, args.skew_s)

    t_start = time.monotonic()
    try:
        while True:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            threading.Thread(
                target=_serve_client,
                args=(conn, addr, args.fps, skew_ns, stop, hmd, t_start),
                name=f"mock-quest-{addr[1]}",
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.close()
        udp_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
