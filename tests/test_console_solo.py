"""[혼자 녹화] timeline (10.12 user: one operator, no assistant), on a fake clock."""

from __future__ import annotations

import pytest

from motion_acq.console import solo
from motion_acq.console.solo import RECORD_READY, RECORD_RECORDING, SoloAbort, SoloIO

READY = RECORD_READY[1]
SAVED = "저장됨: home 으로 복귀"
HOMING = "home 으로 복귀 중"


class Rig:
    """Recorder, neck and quest-view as the run sees them."""

    def __init__(self, *, fronts=(True,), head=True, cut_at=None, stop_at=None) -> None:
        self.t = 0.0
        self.phase: str | None = READY
        self.saved_at: float | None = None
        self.head = True if head else None  # locked
        self.fronts = list(fronts)
        self.cut_at = cut_at
        self.stop_at = stop_at
        self.said: list[tuple[float, str]] = []
        self.sent: list[tuple[float, str, str]] = []
        self.front_at: list[float] = []

    def io(self) -> SoloIO:
        return SoloIO(say=lambda text: self.said.append((round(self.t, 1), text)), now=lambda: self.t,
                      sleep=self.sleep, record_phase=self.record_phase, send=self.send,
                      set_front=self.set_front, head_locked=lambda: self.head,
                      stop_requested=lambda: self.stop_at is not None and self.t >= self.stop_at,
                      status=lambda stage, left: None)

    def sleep(self, dt: float) -> None:
        self.t += max(dt, 0.01)
        if self.cut_at is not None and self.t >= self.cut_at and self.phase == RECORD_RECORDING:
            self.phase = HOMING  # tracking lost: the recorder resets the attempt and goes home
        if self.saved_at is not None and self.t - self.saved_at > 3.0:
            self.phase = READY

    def record_phase(self):
        return self.phase

    def send(self, key: str, text: str) -> None:
        self.sent.append((round(self.t, 1), key, text))
        if key == "head":
            self.head = not self.head
        elif self.phase in RECORD_READY:
            self.phase = RECORD_RECORDING
        elif self.phase == RECORD_RECORDING:
            self.phase, self.saved_at = SAVED, self.t

    def set_front(self):
        self.front_at.append(round(self.t, 1))
        ok = self.fronts.pop(0) if self.fronts else True
        return ok, "정면 저장됨" if ok else "고개가 움직였습니다 (8°): 다시"


def texts(rig: Rig) -> list[str]:
    return [text for _, text in rig.said]


def test_countdown_front_follow_for_the_set_time_then_save_and_home():
    rig = Rig()
    line = solo.run(rig.io(), 60)
    assert rig.front_at == [10.0]  # the front at 0 of the countdown
    assert [s for s in rig.sent if s[2] == " "] == [(10.0, "head", " "), (10.0, "record", " "),
                                                    (70.0, "record", " "), (70.0, "head", " ")]
    heard = texts(rig)
    assert heard[:6] == [solo.INTRO, solo.LOOK, "3", "2", "1", "시작"]
    assert heard[6:12] == ["10초 남았습니다", "5", "4", "3", "2", "1"]
    assert heard[12:] == ["종료. 홈으로 돌아갑니다", "저장했습니다"]
    assert rig.head is True and rig.phase in RECORD_READY
    assert "60 초 녹화" in line
    look = dict((text, t) for t, text in rig.said)
    assert look[solo.LOOK] == pytest.approx(5.0)  # 5 s of looking at the front before 0


def test_a_moved_head_looks_at_the_front_again():
    rig = Rig(fronts=(False, True))
    solo.run(rig.io(), 20)
    assert len(rig.front_at) == 2
    assert "정면을 다시 맞춥니다" in texts(rig)
    assert rig.sent[0][1:] == ("head", " ") and rig.sent[0][0] == rig.front_at[-1]


def test_no_start_when_the_front_cannot_be_set():
    rig = Rig(fronts=(False, False, False))
    with pytest.raises(SoloAbort, match="정면"):
        solo.run(rig.io(), 20)
    assert rig.sent == [] and rig.phase in RECORD_READY  # nothing moved


def test_a_recorder_that_stops_recording_ends_the_run_and_sends_the_neck_home():
    rig = Rig(cut_at=30.0)
    with pytest.raises(SoloAbort, match="녹화가 멈췄"):
        solo.run(rig.io(), 60)
    assert [s[1] for s in rig.sent] == ["head", "record", "head"]  # no save Space after a cut
    assert rig.head is True
    assert "문제가 생겨 멈춥니다. 홈으로 돌아갑니다" in texts(rig)


def test_stop_during_the_run_saves_and_goes_home():
    rig = Rig(stop_at=40.0)
    line = solo.run(rig.io(), 60)
    assert [s[1] for s in rig.sent] == ["head", "record", "record", "head"]
    assert "일찍" in line


def test_stop_during_the_countdown_starts_nothing():
    rig = Rig(stop_at=4.0)
    with pytest.raises(SoloAbort, match="중지"):
        solo.run(rig.io(), 60)
    assert rig.sent == []


def test_a_station_without_a_neck_runs_the_arms_only():
    rig = Rig(head=False)
    solo.run(rig.io(), 15)
    assert {s[1] for s in rig.sent} == {"record"}


def test_a_following_neck_is_not_toggled_into_a_lock():
    rig = Rig()
    rig.head = False  # unlocked: Space would lock it, the opposite of what the run wants
    with pytest.raises(SoloAbort, match="목"):
        solo.run(rig.io(), 15)
    assert rig.sent == []


def test_wait_ready_gives_up_on_a_recorder_that_exited():
    rig = Rig()
    rig.phase = None
    with pytest.raises(SoloAbort, match="꺼졌"):
        solo.wait_ready(rig.io())
