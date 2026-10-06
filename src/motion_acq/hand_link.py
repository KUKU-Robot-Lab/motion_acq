"""Arm process <-> RH56F1 hand nodes over localhost UDP: the hand is a fist while the arm moves.

10.06 user: on the arm start the hands go fist -> the arm plays its stored path rest -> home -> the hands
open and follow the gloves; on the stop the hands close again before the path home -> rest. The stored
paths were planned with the RH56F1 closed (sim2real hand_path_pose; an open hand at rest touches the base
plate). The arm process has no ROS, so it talks to each hand node (motion_acq_hand hand_node,
parameter arm_link_port) over UDP:

    arm  -> hand port   {"arm": "path" | "home" | "rest"}     (repeated every beat_s by a heartbeat)
    hand -> arm sender  {"side": .., "at_rest": bool, "mode": .., "phase": .., "fault": ..}

require_rest() blocks the arm until every hand of the moving arms reports at_rest, and raises
HandLinkError when a hand does not answer (no hand node: nothing would close it) or does not reach the
fist in time (something in the hand, a fault): the arm must not start its path then.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterable, Mapping

log = logging.getLogger(__name__)

ARM_PHASES = ("path", "home", "rest")


class HandLinkError(RuntimeError):
    pass


class HandLink:
    def __init__(self, ports: Mapping[str, int], *, wait_s: float = 15.0, answer_s: float = 2.0,
                 beat_s: float = 0.2, required: bool = True, host: str = "127.0.0.1",
                 clock: Callable[[], float] = time.monotonic) -> None:
        """required False (fake robot): a hand node that does not answer is skipped with a warning."""
        self.ports = {str(side): int(port) for side, port in ports.items()}
        self.required = bool(required)
        self.wait_s, self.answer_s, self.beat_s = float(wait_s), float(answer_s), float(beat_s)
        self.host, self.clock = host, clock
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.setblocking(False)
        self.replies: dict[str, tuple[dict, float]] = {}
        self._phase: str | None = None
        self._sides: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    def _send(self, phase: str, sides: Iterable[str]) -> None:
        payload = json.dumps({"arm": phase}).encode()
        for side in sides:
            try:
                self.sock.sendto(payload, (self.host, self.ports[side]))
            except OSError as exc:  # nobody listening is only seen as a missing reply
                log.debug("hand link %s: %s", side, exc)

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            try:
                reply = json.loads(data.decode())
                side = str(reply["side"])
            except (ValueError, KeyError, UnicodeDecodeError):
                continue
            if side in self.ports:
                with self._lock:
                    self.replies[side] = (reply, self.clock())

    def announce(self, phase: str, sides: Iterable[str]) -> None:
        """Tell the hands of these arms what the arm does now; the heartbeat repeats it."""
        if phase not in ARM_PHASES:
            raise ValueError(f"arm phase must be one of {ARM_PHASES}")
        sides = tuple(s for s in sides if s in self.ports)
        with self._lock:
            self._phase, self._sides = phase, sides
        self._send(phase, sides)
        if self._beat is None:
            self._beat = threading.Thread(target=self._heartbeat, name="hand-link", daemon=True)
            self._beat.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.beat_s):
            with self._lock:
                phase, sides = self._phase, self._sides
            if phase is not None:
                self._send(phase, sides)
            self.poll()

    def require_rest(self, sides: Iterable[str]) -> None:
        """Hold the hands of these arms at the fist and wait until each one reports it is there."""
        sides = tuple(s for s in sides if s in self.ports)
        if not sides:
            return
        start = self.clock()
        with self._lock:
            for side in sides:
                self.replies.pop(side, None)
        self.announce("path", sides)
        log.info("hand link: closing the %s hand(s) to the fist before the arm path", "/".join(sides))
        while True:
            self._send("path", sides)
            time.sleep(min(self.beat_s, 0.05))
            self.poll()
            now = self.clock()
            with self._lock:
                got = {s: self.replies.get(s) for s in sides}
            silent = [s for s, r in got.items() if r is None]
            if silent and now - start > self.answer_s and not self.required:
                log.warning("hand link: no %s hand node answers (fake robot): the arm moves without it", silent)
                sides = tuple(s for s in sides if s not in silent)
                got = {s: got[s] for s in sides}
                silent = []
                if not sides:
                    return
            if silent and now - start > self.answer_s:
                raise HandLinkError(f"{'/'.join(silent)} 손 노드가 응답하지 않는다(UDP {self._ports_of(silent)}): "
                                    "손을 주먹으로 쥘 수 없어 팔 경로를 시작하지 않는다. 콘솔에서 로봇 손 노드를 띄울 것")
            if not silent and all(r is not None and r[0].get("at_rest") for r in got.values()):
                log.info("hand link: %s at the fist after %.1f s", "/".join(sides), now - start)
                return
            faults = {s: r[0].get("fault") for s, r in got.items() if r is not None and r[0].get("fault")}
            if faults or now - start > self.wait_s:
                state = {s: (r[0] if r else None) for s, r in got.items()}
                raise HandLinkError(f"손이 {self.wait_s:g} s 안에 주먹이 되지 않았다(손 안에 물건? 손 오류?): {state}")

    def _ports_of(self, sides: Iterable[str]) -> str:
        return ", ".join(f"{s} {self.ports[s]}" for s in sides)

    def close(self) -> None:
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=1.0)
        self.sock.close()


def hand_link_reply(side: str, *, at_rest: bool, mode: str, phase: str | None, fault: str | None) -> bytes:
    """The hand node's answer to an arm message."""
    return json.dumps({"side": side, "at_rest": bool(at_rest), "mode": mode, "phase": phase,
                       "fault": fault}).encode()


def parse_arm_message(data: bytes) -> str | None:
    """The arm phase in a message to a hand node, None if it is not one."""
    try:
        phase = json.loads(data.decode()).get("arm")
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None
    return phase if phase in ARM_PHASES else None
