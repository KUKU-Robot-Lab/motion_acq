"""direct (Nova2Dex, 10.11 user): the SenseGlove SDK's calibration is the only glove calibration.

The hand node runs without a calibration file, the console reads each glove's SDK calibration
from the glove driver's log, reads new steps aloud, and [장갑 보정 다시] resets it.
"""

from __future__ import annotations

import dataclasses

import pytest

from motion_acq.console import probes, units
from motion_acq.hand.direct import sdk_calibration_state
from motion_acq.hand.retarget import HandRetargeter, load_hand_retarget_config
from motion_acq.hand.rh56f1 import load_rh56f1_map

CONFIG = load_hand_retarget_config()


def test_direct_runs_without_a_calibration_file():
    config = dataclasses.replace(CONFIG, method="direct")
    rt = HandRetargeter(config, None, load_rh56f1_map(), "right")
    assert rt.calibration is None and rt.direct is not None
    assert rt.flexion is None and rt.joystick is None and rt.kinematic is None


@pytest.mark.parametrize("method", ["flexion", "examples", "kinematic"])
def test_the_other_methods_still_need_one(method):
    config = dataclasses.replace(CONFIG, method=method)
    with pytest.raises(ValueError, match="needs a calibration"):
        HandRetargeter(config, None, load_rh56f1_map(), "right")


@pytest.mark.parametrize(("text", "known", "done"), [
    (None, False, False),
    ("", False, False),
    ("Move Fingers | Calibration State: Move Fingers", True, False),
    ("All Sensors Moved | Calibration State: All Sensors Moved", True, True),
    ("Calibration Locked | Calibration State: Calibration Locked", True, True),
    ("Unknown | whatever", True, False),
])
def test_sdk_state_for_the_console(text, known, done):
    state = sdk_calibration_state(text)
    assert (state["known"], state["done"]) == (known, done)
    assert state["label"]


def _log(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("".join(f"[nova2_sdk-11] \x1b[0m[INFO]: {line}\x1b[0m\n" for line in lines))
    return path


def test_the_driver_log_gives_each_gloves_last_state(tmp_path):
    _log(tmp_path, "driver_20261011_200000.log", ["right calibration: Calibration Locked | x"])
    newest = _log(tmp_path, "driver_20261011_225357.log", [
        "right calibration: Move Fingers | Calibration State: Move Fingers",
        "left calibration: Move Fingers | Calibration State: Move Fingers",
        "left: calibration reset",  # the command echo is not a state
        "left calibration: All Sensors Moved | Calibration State: All Sensors Moved",
    ])
    path, states = probes.sdk_calibration_from_log(tmp_path)
    assert path == str(newest)
    assert states == {"right": "Move Fingers | Calibration State: Move Fingers",
                      "left": "All Sensors Moved | Calibration State: All Sensors Moved"}


def test_no_driver_log_no_state(tmp_path):
    assert probes.sdk_calibration_from_log(tmp_path) == (None, {})


def _speaker(monkeypatch, mode="real"):
    import motion_acq.utils.speech as speech

    spoken: list[str] = []
    monkeypatch.setattr(speech, "say", lambda text, **kw: spoken.append(text))
    p = object.__new__(probes.Probes)
    p.mode_getter = lambda: mode
    p._sdk_heard = None
    return p, spoken


def _gloves(log, **states):
    return {"sdk_log": log, "gloves": {side: {"sdk_calibration": s} for side, s in states.items()}}


def test_the_console_reads_new_steps_aloud_once(monkeypatch):
    p, spoken = _speaker(monkeypatch)
    p._speak_sdk_calibration(_gloves("a.log", left="All Sensors Moved | x"))  # already there: silent
    assert spoken == []
    p._speak_sdk_calibration(_gloves("b.log", left="Move Fingers | x"))  # a new driver: starts over
    p._speak_sdk_calibration(_gloves("b.log", left="Move Fingers | x"))
    p._speak_sdk_calibration(_gloves("b.log", left="All Sensors Moved | x"))
    p._speak_sdk_calibration(_gloves("b.log", left="Calibration Locked | x"))
    assert len(spoken) == 2
    assert spoken[0].startswith("왼손 장갑 보정.") and spoken[1] == "왼손 장갑 보정 완료"


def test_fake_mode_stays_silent(monkeypatch):
    p, spoken = _speaker(monkeypatch, mode="fake")
    p._speak_sdk_calibration(_gloves("a.log", right=None))
    p._speak_sdk_calibration(_gloves("b.log", right="Move Fingers | x"))
    assert spoken == []


def _station():
    return units.Station.load("arm4090")


def test_the_real_direct_hand_node_gets_no_calibration_file(monkeypatch, tmp_path):
    monkeypatch.setattr(units, "hand_method", lambda: "direct")
    monkeypatch.setattr("motion_acq.hand.nova2.load_gloves",
                        lambda: {s: type("G", (), {"topic": f"/glove_{s}"})() for s in ("left", "right")})
    launch = units.hand(_station(), "real", "right", "", tmp_path)
    assert "calibration:=" not in " ".join(launch.argv)
    monkeypatch.setattr(units, "hand_method", lambda: "flexion")
    with pytest.raises(units.UnitError):  # the older methods still need the pilot's file
        units.hand(_station(), "real", "right", "", tmp_path)


def test_recalibrate_resets_the_sdk_calibration_of_one_glove():
    launch = units.glove_recalibrate(_station(), "real", "left")
    command = " ".join(launch.argv)
    assert "/motion_acq/nova2_sdk/left/calibration_cmd" in command and "{data: reset}" in command
    assert "-t 3" in command and not launch.moves_robot
    with pytest.raises(units.UnitError):
        units.glove_recalibrate(_station(), "fake", "left")
