"""Which adb device is the Quest: the USB cable, or the same headset over Wi-Fi (10.12 user).

``scripts/quest_usb.sh wifi`` (cable in) switches the headset's adb to TCP 5555, connects to it over
Wi-Fi and remembers the address in WIFI_FILE; after that the cable may come out (or stay in to
charge). While remembered, the Wi-Fi device is used (dialled again when adb lost it); otherwise
the USB one. ``wifi-off`` forgets it. Every adb call of motion_acq sets ANDROID_SERIAL from
``pick()`` (adb refuses to guess when two devices are listed: the same headset on USB and Wi-Fi).

Per transport state: adb reverse (the page's http://localhost:8787) and forward (65432, devtools
9333) belong to one device, so the Wi-Fi switch sets them again on the Wi-Fi device.

Over Wi-Fi the poses and the head camera share the room's network: more latency and jitter than
USB, and a dropped link stops the arms (tracking lost). The headset also runs on its battery.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

WIFI_FILE = Path(os.environ.get("MACQ_QUEST_WIFI_FILE", Path.home() / ".config" / "macq" / "quest_wifi"))
WIFI_PORT = 5555
RECONNECT_EVERY_S = 10.0  # an unreachable address makes adb connect wait: not on every probe
_last_connect = 0.0


@dataclass(frozen=True)
class Device:
    serial: str
    state: str  # device / unauthorized / offline / no permissions ...

    @property
    def wifi(self) -> bool:
        return ":" in self.serial  # adb names a TCP device ip:port


def parse_devices(text: str) -> list[Device]:
    """``adb devices`` output -> devices (header and daemon lines skipped)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith(("*", "List of devices")):
            out.append(Device(parts[0], parts[1]))
    return out


def saved_wifi() -> str | None:
    try:
        value = WIFI_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def pick(devices: list[Device], wifi: str | None = None) -> Device | None:
    """The Quest to talk to: the remembered Wi-Fi device when it is up (once switched, everything stays
    on Wi-Fi, cable in or not: a reverse / forward lives on one transport), else a USB device, else any
    ready one; an unready device (unauthorized ...) when nothing is ready, so its state can be reported."""
    ready = [d for d in devices if d.state == "device"]
    for device in ready:
        if wifi and device.serial == wifi:
            return device
    usb = [d for d in ready if not d.wifi]
    if usb:
        return usb[0]
    tcp = [d for d in ready if d.wifi]
    if tcp:
        return tcp[0]
    return devices[0] if devices else None


def adb_path() -> str:
    from motion_acq.quest_view.devtools import adb_path as _adb

    return _adb()


def current(*, reconnect: bool = True, timeout: float = 4.0) -> Device | None:
    """The Quest now; with ``reconnect`` an adb that lost the remembered Wi-Fi device dials it once."""
    adb = adb_path()

    def listed() -> list[Device]:
        try:
            done = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return []
        return parse_devices(done.stdout)

    global _last_connect
    wifi = saved_wifi()
    devices = listed()
    if (reconnect and wifi and not any(d.serial == wifi and d.state == "device" for d in devices)
            and time.monotonic() - _last_connect >= RECONNECT_EVERY_S):
        _last_connect = time.monotonic()
        try:
            subprocess.run([adb, "connect", wifi], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            pass
        devices = listed()
    return pick(devices, wifi)


def adb_env(device: Device | None = None) -> dict[str, str]:
    """The environment for an adb call: ANDROID_SERIAL = the picked Quest (unchanged without one)."""
    env = dict(os.environ)
    device = device if device is not None else current()
    if device is not None:
        env["ANDROID_SERIAL"] = device.serial
    return env
