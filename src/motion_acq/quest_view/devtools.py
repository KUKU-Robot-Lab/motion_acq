"""Start the headset VR view from the PC (no touch in the headset).

    python -m motion_acq.quest_view.devtools            # enter VR on the open page
    python -m motion_acq.quest_view.devtools --status   # print the page state only

WebXR only starts a session from a user gesture inside the page. The Quest
Browser exposes its devtools on the abstract socket chrome_devtools_remote;
through ``adb forward`` the PC evaluates ``macqStartVr()`` in the quest-view
page with ``userGesture: true`` (Chrome DevTools Protocol, Runtime.evaluate),
which counts as that gesture. Used by ``scripts/quest_usb.sh view|vr``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEVTOOLS_SOCKET = "localabstract:chrome_devtools_remote"


def adb_path() -> str:
    return shutil.which("adb") or str(Path.home() / "opt/platform-tools/adb")


def forward(local_port: int) -> None:
    subprocess.run([adb_path(), "forward", f"tcp:{local_port}", DEVTOOLS_SOCKET],
                   check=True, capture_output=True, text=True)


def find_page(local_port: int, view_port: int, timeout_s: float) -> str:
    """websocket debugger URL of the quest-view page, waiting for it to load."""
    deadline = time.monotonic() + timeout_s
    last = "no answer from the browser devtools"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{local_port}/json", timeout=2) as reply:
                pages = json.load(reply)
            for page in pages:
                if page.get("type") == "page" and f"localhost:{view_port}" in page.get("url", ""):
                    return page["webSocketDebuggerUrl"]
            last = "quest-view page not open: " + ", ".join(p.get("url", "?") for p in pages)
        except (OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(0.5)
    raise SystemExit(f"cannot reach the page: {last}")


def command(ws_url: str, method: str, params: dict) -> dict:
    """One devtools command, its result."""
    from websockets.sync.client import connect

    with connect(ws_url, max_size=None, open_timeout=5) as ws:
        ws.send(json.dumps({"id": 1, "method": method, "params": params}))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            message = json.loads(ws.recv(timeout=15))
            if message.get("id") == 1:
                return message.get("result", {})
    raise SystemExit(f"no reply to {method}")


def evaluate(ws_url: str, expression: str, *, user_gesture: bool) -> object:
    from websockets.sync.client import connect

    with connect(ws_url, max_size=None, open_timeout=5) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
            "userGesture": user_gesture}}))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            message = json.loads(ws.recv(timeout=15))
            if message.get("id") == 1:
                result = message.get("result", {})
                if "exceptionDetails" in result:
                    raise SystemExit(f"page error: {result['exceptionDetails'].get('text')}")
                return result.get("result", {}).get("value")
    raise SystemExit("no reply from the page")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="only print window.macqView")
    ap.add_argument("--view-port", type=int, default=8787)
    ap.add_argument("--local-port", type=int, default=9333, help="PC port for the devtools forward")
    ap.add_argument("--wait-s", type=float, default=20.0, help="how long to wait for the page")
    args = ap.parse_args(argv)
    forward(args.local_port)
    url = find_page(args.local_port, args.view_port, args.wait_s)
    state = "JSON.stringify(window.macqView || null)"
    if args.status:
        print(evaluate(url, state, user_gesture=False))
        return 0
    if evaluate(url, "typeof window.macqStartVr", user_gesture=False) != "function":
        # an old cached viewer.js: reload past the cache once
        command(url, "Page.reload", {"ignoreCache": True})
        time.sleep(2.0)
        url = find_page(args.local_port, args.view_port, args.wait_s)
    deadline = time.monotonic() + args.wait_s
    while evaluate(url, "typeof window.macqStartVr", user_gesture=False) != "function":
        if time.monotonic() > deadline:
            raise SystemExit("the page has no macqStartVr() even after a reload; is quest-view up to date?")
        time.sleep(0.5)
    result = evaluate(url, "window.macqStartVr()", user_gesture=True)
    print(f"VR: {result}; page state {evaluate(url, state, user_gesture=False)}")
    return 0 if result in ("started", "already") else 1


if __name__ == "__main__":
    sys.exit(main())
