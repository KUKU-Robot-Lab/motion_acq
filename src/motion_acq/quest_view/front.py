"""The robot's front for one VR session: set once, shared by both arms and the neck.

10.11 (arm tuning): each arm took robot forward from the headset heading at its
own Space press. The right arm started while the operator looked at the console
(-29 deg) and then turned to the robot (-94 deg): its hand moved 65 deg off,
diagonally on the robot. With the neck following the head the gaze at a Space
press says even less. Now the operator faces the robot's front once, the console
asks quest-view to keep that headset pose, and every pose frame carries it
("front"); the arms and the neck take forward from it.

The page's reference space (local-floor) is new at every VR start and moves on
a Quest recenter, so a front belongs to one page session id (the page makes a
new one at both) and is dropped when the id changes.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from motion_acq.quest_view.convert import unity_position, unity_quaternion

STILL_S = 1.0  # the headset must hold its direction this long
STILL_DEG = 5.0  # yaw and pitch spread allowed over STILL_S
FRESH_S = 0.3  # newest pose at most this old
MIN_SAMPLES = 20  # poses within STILL_S (the page sends one per XR frame, 72 Hz)


def look_deg(q: Any) -> tuple[float, float]:
    """(yaw, pitch) in deg of a WebXR orientation [x, y, z, w]: where -z (the gaze) points.

    yaw > 0 turns left (about +y), pitch > 0 looks up."""
    x, y, z, w = (float(v) for v in q)
    # forward = R @ (0, 0, -1)
    fx = -(2.0 * (x * z + w * y))
    fy = -(2.0 * (y * z - w * x))
    fz = -(1.0 - 2.0 * (x * x + y * y))
    return math.degrees(math.atan2(-fx, -fz)), math.degrees(math.atan2(fy, math.hypot(fx, fz)))


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Front:
    session: str
    position: tuple[float, float, float]  # WebXR
    rotation: tuple[float, float, float, float]  # WebXR [x, y, z, w]
    yaw_deg: float
    pitch_deg: float
    set_at: float

    def wire(self) -> dict[str, dict[str, float]]:
        """HandUMI wire form (raw Unity coordinates), like hmdPosition / hmdRotation."""
        return {"position": unity_position(self.position), "rotation": unity_quaternion(self.rotation)}


class FrontKeeper:
    """Recent headset poses of the page, and the front the console set from them."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self.session: str | None = None
        self.front: Front | None = None
        self.dropped: str | None = None  # why the last front went away (for the console)
        self._recent: deque[tuple[float, tuple, tuple]] = deque()

    def observe(self, message: dict) -> None:
        """One page pose message ({"session", "hmd": {"p", "q"}})."""
        now = self.clock()
        session = message.get("session")
        session = str(session) if isinstance(session, str) and session else None
        if session != self.session:
            self._recent.clear()
            self.session = session
        if self.front is not None and session != self.front.session:
            self.front = None
            self.dropped = "VR 이 다시 시작되었거나 Quest 가 재중심되어 정면이 풀렸습니다"
        hmd = message.get("hmd")
        if isinstance(hmd, dict):
            try:
                p = tuple(float(v) for v in hmd["p"])
                q = tuple(float(v) for v in hmd["q"])
            except (KeyError, TypeError, ValueError):
                return
            if len(p) == 3 and len(q) == 4 and all(map(math.isfinite, (*p, *q))):
                self._recent.append((now, p, q))
        while self._recent and now - self._recent[0][0] > STILL_S:
            self._recent.popleft()

    def set(self) -> tuple[bool, str]:
        """Keep the headset pose of now as the front. (ok, Korean text for the operator)."""
        now = self.clock()
        if self.session is None:
            return False, "헤드셋 페이지가 VR 중이 아닙니다 (VR 시작 먼저, 안 되면 페이지 다시 열기)"
        if not self._recent or now - self._recent[-1][0] > FRESH_S:
            return False, "헤드셋 자세가 들어오지 않습니다 (헤드셋을 쓰고 VR 시작)"
        if len(self._recent) < MIN_SAMPLES or now - self._recent[0][0] < 0.8 * STILL_S:
            return False, "헤드셋 자세가 아직 모자랍니다: 1 초 뒤 다시"
        looks = [look_deg(q) for _, _, q in self._recent]
        yaw0 = looks[-1][0]
        yaw_spread = max(abs(_wrap(y - yaw0)) for y, _ in looks)
        pitch_spread = max(p for _, p in looks) - min(p for _, p in looks)
        if max(yaw_spread, pitch_spread) > STILL_DEG:
            return False, f"고개가 움직였습니다 ({max(yaw_spread, pitch_spread):.0f}°): 로봇 정면을 보고 1 초 가만히 있다가 다시"
        _, p, q = self._recent[-1]
        yaw, pitch = looks[-1]
        self.front = Front(self.session, p, q, yaw, pitch, now)
        self.dropped = None
        return True, f"정면 저장됨 (헤드셋 yaw {yaw:+.0f}°, pitch {pitch:+.0f}°)"

    def clear(self) -> None:
        self.front = None
        self.dropped = None

    def wire(self) -> dict | None:
        return None if self.front is None else self.front.wire()

    def status(self) -> dict[str, Any]:
        front = self.front
        return {
            "set": front is not None,
            "yaw_deg": None if front is None else round(front.yaw_deg, 1),
            "pitch_deg": None if front is None else round(front.pitch_deg, 1),
            "age_s": None if front is None else round(self.clock() - front.set_at, 1),
            "vr": self.session is not None,
            "dropped": self.dropped,
        }
