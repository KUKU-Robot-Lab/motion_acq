"""The robot's front: set once per VR session, shared by both arms and the neck (10.11).

10.11 23:25: the right arm anchored while the operator looked at the console
(headset -29 deg), then turned to the robot (-94 deg); the hand moved about 65 deg
off, so X / Y tracked diagonally on the robot while the left arm (anchored facing
the robot) followed well.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from head_helpers import make_config, pose_from_yaw_pitch
from test_quest_view import free_port, webxr_quat
from test_teleop_heading import OPEN, TRACKED, Sample, Widths, controller, yaw_pose7

from motion_acq.head.retarget import AxisConfig, HeadRetargeter
from motion_acq.quest_view.convert import handumi_frame
from motion_acq.quest_view.front import STILL_S, FrontKeeper, look_deg
from motion_acq.robots.registry import load_embodiment
from motion_acq.teleop.common import announce_front_missing
from motion_acq.teleop.session import TeleopSession


@pytest.fixture(scope="module")
def runtime():
    return load_embodiment("openarmv1")


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def page(session: str | None, yaw: float = 0.0, pitch: float = 0.0) -> dict:
    return {"type": "pose", "session": session, "hmd": {"p": [0.0, 1.5, 0.0], "q": webxr_quat(yaw, pitch)}}


def feed(keeper: FrontKeeper, clock: Clock, seconds: float, *, session="s1", yaw=0.0, wobble=0.0) -> None:
    for i in range(int(seconds * 72)):
        clock.t += 1 / 72
        keeper.observe(page(session, yaw + wobble * math.sin(i / 5.0)))


@pytest.mark.parametrize("yaw, pitch", [(0.0, 0.0), (40.0, -10.0), (-120.0, 15.0)])
def test_look_angles_of_a_webxr_orientation(yaw, pitch):
    got = look_deg(webxr_quat(yaw, pitch))
    assert got == pytest.approx((yaw, pitch), abs=1e-6)


def test_the_front_needs_vr_a_fresh_pose_and_a_still_head():
    clock = Clock()
    keeper = FrontKeeper(clock)
    assert not keeper.set()[0]  # nothing yet
    feed(keeper, clock, 1.5, session=None)
    ok, text = keeper.set()
    assert not ok and "VR" in text  # page not in VR (old page without a session id)
    feed(keeper, clock, 1.5, wobble=20.0)
    ok, text = keeper.set()
    assert not ok and "움직" in text
    feed(keeper, clock, STILL_S + 0.2, yaw=-94.0)
    clock.t += 1.0
    assert not keeper.set()[0]  # poses stopped arriving
    feed(keeper, clock, STILL_S + 0.2, yaw=-94.0)
    ok, text = keeper.set()
    assert ok and keeper.front is not None
    assert keeper.front.yaw_deg == pytest.approx(-94.0, abs=1e-3)
    assert keeper.status()["set"] and keeper.status()["yaw_deg"] == pytest.approx(-94.0)


def test_a_new_vr_session_or_recenter_drops_the_front():
    clock = Clock()
    keeper = FrontKeeper(clock)
    feed(keeper, clock, 1.5)
    assert keeper.set()[0]
    feed(keeper, clock, 0.1)
    assert keeper.front is not None
    feed(keeper, clock, 0.1, session="s2")  # the page made a new id: VR restart or recenter
    assert keeper.front is None and keeper.wire() is None
    assert "풀렸" in keeper.status()["dropped"]


def test_the_wire_front_is_the_headset_pose_in_unity_terms():
    clock = Clock()
    keeper = FrontKeeper(clock)
    feed(keeper, clock, 1.5, yaw=30.0)
    assert keeper.set()[0]
    frame = handumi_frame(page("s1", 30.0))
    assert keeper.wire() == {"position": frame["hmdPosition"], "rotation": frame["hmdRotation"]}


# ---- arms ---------------------------------------------------------------------------------------


def test_the_arm_takes_forward_from_the_front_not_from_where_the_headset_looks(runtime):
    """The 10.11 right arm: front at 90 deg, headset at the console (0 deg) at Space."""
    ctl = controller(runtime, heading=True)
    session = TeleopSession(ctl, joint_filter=PassFilter())
    start = yaw_pose7([0.4, -0.2, -0.3], 30.0)
    sample = Sample()
    sample.hmd_pose = yaw_pose7([0, 0, 0], 0.0)
    sample.front_required = True
    sample.front_pose = yaw_pose7([0, 0, 0], 90.0)
    sample.left_tcp_pose = sample.right_tcp_pose = start
    inputs = TeleopSession.inputs(sample, Widths())
    assert not inputs.front_missing
    frame = session.advance(inputs, now_s=0.0, start_sides=("right",))
    assert frame.anchored_sides == ("right",)
    moved = start.copy()
    moved[1] += 0.05  # toward the front (90 deg = +y in the workspace)
    target = ctl.step({"left": start, "right": moved}, TRACKED, OPEN).target_pose7["right"]
    np.testing.assert_allclose(target[:3] - ctl.anchor_ref["right"][:3], [0.05, 0.0, 0.0], atol=2e-4)


def test_no_arm_starts_while_the_stream_needs_a_front(runtime, caplog):
    ctl = controller(runtime, heading=True)
    session = TeleopSession(ctl, joint_filter=PassFilter())
    sample = Sample()
    sample.front_required = True
    sample.front_pose = None
    inputs = TeleopSession.inputs(sample, Widths())
    assert inputs.front_missing
    with caplog.at_level("WARNING"):
        frame = session.advance(inputs, now_s=0.0, start_sides=("right",))
    assert frame.anchored_sides == () and ctl.anchors["right"] is None
    assert "정면 맞추기" in caplog.text


def test_a_stream_without_a_front_keeps_the_headset_heading(runtime):
    """The HandUMI app and the mock sender carry no front: the arm starts as before."""
    ctl = controller(runtime, heading=True)
    session = TeleopSession(ctl, joint_filter=PassFilter())
    inputs = TeleopSession.inputs(Sample(), Widths())
    assert not inputs.front_missing and inputs.front_pose7 is None
    assert session.advance(inputs, now_s=0.0, start_sides=("right",)).anchored_sides == ("right",)


def test_a_refused_start_is_spoken(monkeypatch):
    spoken = []
    monkeypatch.setattr("motion_acq.utils.speech.say", lambda text, **kw: spoken.append(text))
    assert announce_front_missing(("right",), (), True, play_sounds=True)
    assert spoken == ["정면 맞추기를 먼저 하세요"]
    assert not announce_front_missing(("right",), ("right",), True, play_sounds=True)
    assert not announce_front_missing((), (), True, play_sounds=True)
    assert not announce_front_missing(("right",), (), False, play_sounds=True)


class PassFilter:
    def filter(self, q, now_s, *, exact_indices=()):
        return q

    def snapshot(self):
        return None

    def restore(self, state):
        pass


# ---- neck ---------------------------------------------------------------------------------------


def test_the_neck_is_home_when_the_headset_looks_at_the_front():
    """Anchored looking 30 deg left of the front: the head walks to home + 30 deg, not home."""
    rt = HeadRetargeter(make_config())
    front = pose_from_yaw_pitch(-60.0, 0.0)
    assert rt.anchor(pose_from_yaw_pitch(-30.0, 0.0), (-2.9, 71.8), 0.0, front_pose7=front)
    step = rt.step(pose_from_yaw_pitch(-30.0, 0.0), True, 0.02)
    assert step.rel_yaw_deg == pytest.approx(30.0, abs=1e-6)
    assert step.command_pan_deg == pytest.approx(-2.9 + 20.0)  # clamped to the 20 deg window
    step = rt.step(pose_from_yaw_pitch(-60.0, 0.0), True, 0.04)
    assert step.command_pan_deg == pytest.approx(-2.9, abs=1e-3)


def test_with_a_home_pitch_the_tilt_follows_the_absolute_head_pitch():
    """Option B (10.12): home looks 18.9 deg down; the front's own pitch does not matter."""
    rt = HeadRetargeter(make_config(
        tilt=AxisConfig(home_deg=71.8, range_deg=15.0, range_pos_deg=30.0, sign=-1.0),
        tilt_home_pitch_deg=-18.9))
    front = pose_from_yaw_pitch(0.0, -5.0)  # taken while looking a little down
    assert rt.anchor(pose_from_yaw_pitch(0.0, 0.0), (-2.9, 71.8), 0.0, front_pose7=front)
    step = rt.step(pose_from_yaw_pitch(0.0, 0.0), True, 0.02)  # looking level
    assert step.rel_pitch_deg == pytest.approx(18.9, abs=1e-6)
    assert step.command_tilt_deg == pytest.approx(71.8 - 18.9, abs=1e-3)  # camera level
    step = rt.step(pose_from_yaw_pitch(0.0, -18.9), True, 0.04)
    assert step.command_tilt_deg == pytest.approx(71.8, abs=1e-3)  # home = the table view
    step = rt.step(pose_from_yaw_pitch(0.0, -30.0), True, 0.06)
    assert step.command_tilt_deg == pytest.approx(71.8 + 11.1, abs=1e-3)
    step = rt.step(pose_from_yaw_pitch(0.0, -60.0), True, 0.08)
    assert step.command_tilt_deg == pytest.approx(71.8 + 15.0)  # the down window (34 deg below level)


def test_below_level_the_camera_tilts_by_the_down_gain():
    """10.12 user: the solo run looked at most 36 deg down, the table needs ~73: down x2, up 1:1."""
    rt = HeadRetargeter(make_config(
        tilt=AxisConfig(home_deg=71.8, range_deg=15.0, range_pos_deg=75.8, sign=-1.0),
        tilt_home_pitch_deg=-72.8, tilt_down_gain=2.0))
    assert rt.anchor(pose_from_yaw_pitch(0.0, 0.0), (-2.9, 71.8), 0.0, front_pose7=pose_from_yaw_pitch(0.0, 0.0))
    step = rt.step(pose_from_yaw_pitch(0.0, 0.0), True, 0.02)  # level stays level
    assert step.command_tilt_deg == pytest.approx(71.8 - 72.8, abs=1e-3)
    step = rt.step(pose_from_yaw_pitch(0.0, -18.0), True, 0.04)  # 18 down -> camera 36 down
    assert step.command_tilt_deg == pytest.approx(71.8 - 72.8 + 36.0, abs=1e-3)
    step = rt.step(pose_from_yaw_pitch(0.0, -36.4), True, 0.06)  # 36.4 down -> home, the table view
    assert step.command_tilt_deg == pytest.approx(71.8, abs=1e-3)
    rt.step(pose_from_yaw_pitch(0.0, -10.0), True, 0.08)  # (a 30 deg jump would re-anchor)
    step = rt.step(pose_from_yaw_pitch(0.0, 2.0), True, 0.10)  # up stays 1:1
    assert step.command_tilt_deg == pytest.approx(71.8 - 72.8 - 2.0, abs=1e-3)


def test_the_down_gain_needs_a_robot_front():
    """Without a front (anchor-relative mode) the gain is not applied."""
    rt = HeadRetargeter(make_config(tilt_home_pitch_deg=-72.8, tilt_down_gain=2.0))
    assert rt.anchor(pose_from_yaw_pitch(0.0, 0.0), (0.0, 0.0), 0.0)
    step = rt.step(pose_from_yaw_pitch(0.0, -10.0), True, 0.02)
    assert step.rel_pitch_deg == pytest.approx(-10.0, abs=1e-6)


def test_the_down_gain_is_bounded():
    from motion_acq.head.retarget import RetargetConfig

    with pytest.raises(ValueError):
        RetargetConfig(pan=AxisConfig(home_deg=0.0, range_deg=20.0), tilt=AxisConfig(home_deg=0.0, range_deg=20.0),
                       tilt_down_gain=0.5)


def test_the_station_neck_uses_the_absolute_pitch():
    import yaml

    from motion_acq.head.config import head_config_from_section

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "stations" / "arm4090.yaml").read_text())["head"]
    config = head_config_from_section(data)
    assert config.retarget.tilt_home_pitch_deg == pytest.approx(-72.8)  # measured 10.12
    assert config.retarget.tilt_down_gain == pytest.approx(2.0)  # 10.12 user
    assert config.tilt_window[0] < config.retarget.tilt.home_deg < config.tilt_window[1]


def test_a_neck_re_anchor_keeps_the_front():
    rt = HeadRetargeter(make_config())
    front = pose_from_yaw_pitch(0.0, 0.0)
    assert rt.anchor(pose_from_yaw_pitch(0.0, 0.0), (-2.9, 71.8), 0.0, front_pose7=front)
    rt.step(pose_from_yaw_pitch(10.0, 0.0), True, 0.02)
    rt.step(None, False, 0.04)  # HMD lost ...
    step = rt.step(pose_from_yaw_pitch(10.0, 0.0), True, 1.0)  # ... long enough to re-anchor
    assert step.reanchored
    step = rt.step(pose_from_yaw_pitch(10.0, 0.0), True, 1.02)
    assert step.rel_yaw_deg == pytest.approx(10.0, abs=1e-6)  # still from the front, not 0


def test_the_neck_waits_for_a_front_and_holds_when_it_is_dropped():
    from dataclasses import dataclass

    from test_head_driver import ScriptedTracker, fake_bus, make_session

    @dataclass
    class FrontSample:
        device_hmd_pose: np.ndarray
        hmd_tracked: bool
        device_front_pose: np.ndarray | None
        front_required: bool = True

    look = pose_from_yaw_pitch(10.0, 0.0)
    front = pose_from_yaw_pitch(0.0, 0.0)
    samples = ([FrontSample(look, True, None)] * 3 + [FrontSample(look, True, front)] * 3
               + [FrontSample(look, True, None)] * 2)
    session = make_session(samples, fake_bus())
    assert isinstance(session.tracker, ScriptedTracker)
    anchored = []
    steps = []
    for _ in samples:
        steps.append(session.tick())
        anchored.append(session.anchored)
    assert anchored[:3] == [False] * 3  # waited for a front
    assert anchored[3] and session.retargeter.front is not None
    assert steps[4] is not None and steps[4].rel_yaw_deg == pytest.approx(10.0, abs=1e-6)
    assert steps[-1] is not None and steps[-1].state.value == "hold"  # front dropped: hold


# ---- quest-view end to end ----------------------------------------------------------------------


def test_quest_view_keeps_the_front_and_sends_it_in_every_frame():
    """Page -> quest-view (POST /front) -> HandUMI stream -> provider sample."""
    import websockets.sync.client

    from motion_acq.calibration.control_tcp import ControllerTcpCalibration
    from motion_acq.quest_view.server import parse_args, serve
    from motion_acq.robots.utils import IDENTITY_POSE7
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
    base = f"http://127.0.0.1:{ports['port']}"

    def post_front() -> dict:
        request = urllib.request.Request(f"{base}/front", method="POST", data=b"")
        with urllib.request.urlopen(request, timeout=2) as reply:
            return json.loads(reply.read())

    try:
        time.sleep(0.5)
        config = MetaQuestConfig(quest_ip="127.0.0.1", tcp_port=ports["tcp"], sync_port=ports["sync"])
        identity = np.asarray(IDENTITY_POSE7, dtype=np.float64)
        provider = MetaQuestTrackingProvider(
            config=config, calibration=ControllerTcpCalibration(left=identity.copy(), right=identity.copy()),
            reset_workspace_on_x=False)
        provider.start()
        with websockets.sync.client.connect(f"ws://127.0.0.1:{ports['port']}/ws") as ws:
            def stream(seconds: float, session: str, yaw: float) -> None:
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    ws.send(json.dumps(page(session, yaw)))
                    time.sleep(1 / 72)

            stream(0.5, "s1", 0.0)
            sample = provider.latest()
            assert sample.streaming and sample.front_required and sample.front_pose is None
            stream(STILL_S + 0.3, "s1", 40.0)
            reply = post_front()
            assert reply["ok"] and reply["yaw_deg"] == pytest.approx(40.0, abs=0.5)
            stream(0.3, "s1", 0.0)  # the operator looks elsewhere: the front stays
            sample = provider.latest()
            assert sample.device_front_pose is not None
            from motion_acq.head.retarget import hmd_yaw_pitch_deg

            assert hmd_yaw_pitch_deg(sample.device_front_pose)[0] == pytest.approx(40.0, abs=0.5)
            assert hmd_yaw_pitch_deg(sample.device_hmd_pose)[0] == pytest.approx(0.0, abs=0.5)
            stream(0.3, "s2", 0.0)  # VR restarted
            sample = provider.latest()
            assert sample.front_required and sample.front_pose is None
            with urllib.request.urlopen(f"{base}/front", timeout=2) as reply:
                assert not json.loads(reply.read())["set"]
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
