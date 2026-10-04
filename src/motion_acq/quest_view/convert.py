"""WebXR pose messages -> HandUMI Quest App wire frames (raw Unity coordinates).

WebXR is right-handed (x right, y up, z backward: you look along -z); Unity,
which the HandUMI app sends, is left-handed (x right, y up, z forward).
Mirroring z maps one to the other: position (x, y, -z), quaternion
(-x, -y, z, w). A WebXR turn to the left (+ about y) becomes a Unity turn
of - about y, which is "look left" there too (Unity +yaw looks right).

Page message (one per XR frame):
    {"dt": s, "hmd": {"p": [x,y,z], "q": [x,y,z,w]} | null,
     "left"/"right": {"p": [...], "q": [...], "buttons": [bool,...], "axes": [float,...]} | null}
Buttons follow the WebXR xr-standard gamepad of the Quest Touch controllers:
0 trigger, 1 grip, 3 thumbstick press, 4 A/X, 5 B/Y; axes 2/3 = thumbstick x/y (y down).
"""

from __future__ import annotations

import math
import time
from typing import Any


def unity_position(p: Any) -> dict[str, float]:
    x, y, z = (float(v) for v in p)
    return {"x": x, "y": y, "z": -z}


def unity_quaternion(q: Any) -> dict[str, float]:
    x, y, z, w = (float(v) for v in q)
    return {"x": -x, "y": -y, "z": z, "w": w}


def _pose(entry: Any) -> tuple[list[float], list[float]] | None:
    if not isinstance(entry, dict):
        return None
    p, q = entry.get("p"), entry.get("q")
    try:
        p, q = [float(v) for v in p], [float(v) for v in q]
    except (TypeError, ValueError):
        return None
    if len(p) != 3 or len(q) != 4 or not all(math.isfinite(v) for v in (*p, *q)):
        return None
    return p, q


def _pressed(entry: Any, index: int) -> bool:
    buttons = entry.get("buttons") if isinstance(entry, dict) else None
    return bool(isinstance(buttons, list) and index < len(buttons) and buttons[index])


def _stick(entry: Any) -> dict[str, float]:
    axes = entry.get("axes") if isinstance(entry, dict) else None
    if not isinstance(axes, list) or len(axes) < 4:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {"x": float(axes[2]), "y": -float(axes[3]), "z": 0.0}


def handumi_frame(message: dict, device_time_ns: int | None = None) -> dict:
    """One HandUMI wire frame from one page message (keys as the Quest App sends them)."""
    frame: dict[str, Any] = {
        "ovrTimeNs": int(time.monotonic_ns() if device_time_ns is None else device_time_ns),
        "deltaTime": float(message.get("dt") or 0.0),
    }
    hmd = _pose(message.get("hmd"))
    if hmd is not None:  # absent keys = HMD untracked, as the app does
        frame["hmdPosition"] = unity_position(hmd[0])
        frame["hmdRotation"] = unity_quaternion(hmd[1])
    for side, a, b in (("left", "X", "Y"), ("right", "A", "B")):
        entry = message.get(side)
        pose = _pose(entry)
        tracked = pose is not None
        p, q = pose if pose is not None else ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        frame.update({
            f"{side}ControllerPosition": unity_position(p),
            f"{side}ControllerRotation": unity_quaternion(q),
            f"{side}Tracked": tracked,
            f"{side}Valid": tracked,
            f"{side}Joystick": _stick(entry),
            f"{side}ThumbstickClick": _pressed(entry, 3),
            f"{side}TriggerPressed": _pressed(entry, 0),
            f"{side}GripPressed": _pressed(entry, 1),
            f"button{a}Pressed": _pressed(entry, 4),
            f"button{b}Pressed": _pressed(entry, 5),
        })
    return frame
