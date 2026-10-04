"""What the headset shows before and after the neck starts.

User 10.04: once the Quest is connected the wearer should see the Quest's own front
view (passthrough); the head camera picture only from the neck start on. The page
started an opaque immersive-vr session as soon as it connected, so the wearer was
put in a dark camera screen right away.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from motion_acq.quest_view import server

WEB = Path(server.__file__).resolve().parent / "web"


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/node")))
    return nvm[-1] if nvm else None


NODE = _node()


def _view_mode(expr: str):
    assert NODE is not None
    script = f"const m = require({json.dumps(str(WEB / 'view_mode.js'))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=20, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_passthrough_is_tried_first_then_vr():
    assert _view_mode("m.SESSION_MODES") == ["immersive-ar", "immersive-vr"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_before_the_neck_the_wearer_sees_the_room():
    look = _view_mode('m.frameLook("immersive-ar", false)')
    assert look["camera"] is False
    assert look["clear"][3] == 0  # transparent: the Quest passthrough shows


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_head_camera_frames_cover_the_view():
    look = _view_mode('m.frameLook("immersive-ar", true)')
    assert look["camera"] is True
    assert look["clear"][3] == 1


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_without_passthrough_the_old_vr_screen_stays():
    look = _view_mode('m.frameLook("immersive-vr", false)')
    assert look["camera"] is True
    assert look["clear"][3] == 1


def test_page_loads_view_mode_before_viewer_both_versioned():
    async def fetch() -> str:
        app = server.build_app(server.Shared(), server.PoseBroadcast(), video_fps=30.0)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/")
            assert response.status == 200
            return await response.text()

    html = asyncio.run(fetch())
    mode_at = html.index("/static/view_mode.js?v=")
    viewer_at = html.index("/static/viewer.js?v=")
    assert mode_at < viewer_at


def _viewer_start(scenario: str) -> dict:
    assert NODE is not None
    harness = Path(__file__).resolve().parent / "js" / "viewer_start.js"
    out = subprocess.run([NODE, str(harness), scenario, str(WEB)], capture_output=True, text=True,
                         timeout=20, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_double_start_makes_one_session():
    """Review 10.04: tap + tap (or tap + quest_usb.sh vr) started two sessions, one without a frame loop."""
    run = _viewer_start("double")
    assert sorted(run["results"]) == ["started", "starting"]
    assert run["requested"] == 1 and run["frameLoops"] == 1 and run["active"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_failed_start_ends_its_session_and_can_retry():
    run = _viewer_start("fail")
    assert run["first"].startswith("error") and run["ended"] == 1
    assert run["retry"] == "started" and run["active"] and run["frameLoops"] == 1
