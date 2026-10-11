"""[혼자 녹화]: one operator, no assistant (10.12 user).

Press once: count down 10 s aloud (the headset goes on, the controllers in hand), take the robot's
front at 0 (the operator looks at the robot's front for the last seconds), start the neck and the
arms (the hands follow with them) together, record for a fixed time, then save and send everything
home. A recorder that stops recording on its own (tracking lost, a fault: it returns the arms home
by itself) ends the run early and the neck goes home too. [중지] ends it early and saves.

The timeline only talks to the console through ``SoloIO``, so it is tested with a fake clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

COUNTDOWN_S = 10
LOOK_AT_FRONT_S = 5  # "look at the robot's front" this long before 0 (the front needs 1 s still)
FRONT_TRIES = 3
WARN_LEFT_S = 10
READY_TIMEOUT_S = 240.0  # recorder start: hands, rest -> home
SAVE_TIMEOUT_S = 60.0
TICK_S = 0.1

# record phases (console.phases RECORD_RULES) the run waits for / watches
RECORD_READY = ("준비: Space 로 에피소드 시작", "에피소드 준비: Space 로 시작")
RECORD_RECORDING = "녹화 중: Space 로 저장"
RECORD_NO_CONTROLLERS = "컨트롤러 트래킹 기다림"  # the recorder holds until both controllers are tracked
NO_CONTROLLERS_AFTER_S = 3.0  # a brief gap is normal right after home
NO_CONTROLLERS_EVERY_S = 15.0

PREPARING = "로봇을 준비합니다. 잠시 기다리세요"
NO_CONTROLLERS = "컨트롤러가 보이지 않습니다. 양손 컨트롤러를 켜고 헤드셋 앞에 들어 주세요"
INTRO = "10초 뒤 시작합니다. 헤드셋을 쓰고 컨트롤러를 잡으세요"
LOOK = "로봇 정면을 보고 가만히 계세요"
PHRASES = (PREPARING, NO_CONTROLLERS, INTRO, LOOK, "3", "2", "1", "시작", "10초 남았습니다", "5", "4",
           "종료. 홈으로 돌아갑니다", "저장했습니다", "문제가 생겨 멈춥니다. 홈으로 돌아갑니다",
           "정면을 다시 맞춥니다")


class SoloAbort(RuntimeError):
    """The run stopped before or while following; the text says why (Korean, for the operator)."""


@dataclass
class SoloIO:
    say: Callable[[str], None]
    now: Callable[[], float]
    sleep: Callable[[float], None]
    record_phase: Callable[[], str | None]  # None: the recorder is not running; "" no phase yet
    send: Callable[[str, str], None]  # (unit key, text) to its terminal
    set_front: Callable[[], tuple[bool, str]]  # (ok, text); ok without a front-carrying stream
    head_locked: Callable[[], bool | None]  # None: no neck on this station / not running
    stop_requested: Callable[[], bool]
    status: Callable[[str, float | None], None]  # (stage, seconds left) for the console card


def wait_ready(io: SoloIO, timeout_s: float = READY_TIMEOUT_S) -> None:
    """Until the recorder waits at home for Space (it may still be starting: hands, rest -> home).
    Spoken, since the operator already wears the headset (10.12: a silent 2 min wait for controllers
    that were not tracked): once at the start, and again while the controllers are not seen."""
    deadline = io.now() + timeout_s
    announced = False
    no_controllers_since: float | None = None
    warned_at: float | None = None
    while (phase := io.record_phase()) not in RECORD_READY:
        if phase is None:
            raise SoloAbort("녹화가 꺼졌습니다: 녹화 칸의 로그 확인")
        if io.now() > deadline:
            raise SoloAbort(f"녹화가 {timeout_s:.0f} 초 안에 준비되지 않았습니다 (지금: {phase or '?'})")
        if io.stop_requested():
            raise SoloAbort("중지")
        if not announced:
            io.say(PREPARING)
            announced = True
        if phase == RECORD_NO_CONTROLLERS:
            now = io.now()
            no_controllers_since = now if no_controllers_since is None else no_controllers_since
            due = now - no_controllers_since >= NO_CONTROLLERS_AFTER_S
            if due and (warned_at is None or now - warned_at >= NO_CONTROLLERS_EVERY_S):
                io.say(NO_CONTROLLERS)
                warned_at = now
            io.status("컨트롤러 안 보임", None)
        else:
            no_controllers_since = None
            io.status("준비 중", None)
        io.sleep(TICK_S)


def _until(io: SoloIO, t: float, end: float) -> None:
    """Wait for io.now() >= t (showing the seconds to ``end``), as long as the recorder stays ready."""
    while io.now() < t:
        if io.stop_requested():
            raise SoloAbort("중지")
        phase = io.record_phase()
        if phase not in RECORD_READY:
            raise SoloAbort(f"녹화가 준비 상태가 아닙니다 ({phase})")
        io.status("카운트다운", max(end - io.now(), 0.0))
        io.sleep(min(TICK_S, t - io.now()))


def countdown(io: SoloIO, seconds: int = COUNTDOWN_S) -> None:
    """Count down aloud and take the front at 0; the operator looks at the robot's front from
    LOOK_AT_FRONT_S before 0 (the front needs the head still for 1 s). A refused front looks again."""
    end = io.now() + seconds
    io.say(INTRO)
    _until(io, end - LOOK_AT_FRONT_S, end)
    for attempt in range(FRONT_TRIES):
        io.say(LOOK)
        for n in (3, 2, 1):
            _until(io, end - n, end)
            io.say(str(n))
        _until(io, end, end)
        ok, text = io.set_front()
        if ok:
            return
        if attempt == FRONT_TRIES - 1:
            raise SoloAbort(f"정면을 못 맞췄습니다: {text}")
        io.say("정면을 다시 맞춥니다")
        end = io.now() + LOOK_AT_FRONT_S


def run(io: SoloIO, duration_s: float, *, countdown_s: int = COUNTDOWN_S) -> str:
    """The whole run; returns a line for the console. SoloAbort when it could not start or was cut."""
    wait_ready(io)
    head = io.head_locked()
    if head is False:
        raise SoloAbort("목이 따라가는 중입니다: 목을 잠그고(home) 다시")
    countdown(io, countdown_s)
    if head is not None:
        io.send("head", " ")  # unlock: the neck follows, home = the front just taken
    io.send("record", " ")  # start the episode; the arms anchor (and their hands follow)
    io.say("시작")
    started = io.now()
    end = started + float(duration_s)
    said: set[float] = set()
    reason = None
    while io.now() < end:
        left = end - io.now()
        io.status("녹화 중", left)
        if io.stop_requested():
            reason = "중지"
            break
        phase = io.record_phase()
        if phase != RECORD_RECORDING and io.now() - started > 2.0:
            reason = f"녹화가 멈췄습니다 ({phase})"
            break
        for mark, text in ((WARN_LEFT_S, "10초 남았습니다"), (5, "5"), (4, "4"), (3, "3"), (2, "2"), (1, "1")):
            if left <= mark and mark not in said:
                said.add(mark)
                if left > mark - 0.5:
                    io.say(text)
        io.sleep(min(TICK_S, max(left, 0.0)))
    recorded_s = io.now() - started
    cut = reason is not None and reason != "중지"
    if not cut:
        io.send("record", " ")  # save; the recorder takes the arms home
    io.say("문제가 생겨 멈춥니다. 홈으로 돌아갑니다" if cut else "종료. 홈으로 돌아갑니다")
    if head is not None and io.head_locked() is False:
        io.send("head", " ")  # lock: the neck walks home
    if cut:
        raise SoloAbort(reason or "")
    deadline = io.now() + SAVE_TIMEOUT_S
    while io.record_phase() not in RECORD_READY:
        io.status("저장·홈 복귀", None)
        if io.record_phase() is None or io.now() > deadline:
            raise SoloAbort(f"저장 뒤 홈 대기로 돌아오지 않았습니다 ({io.record_phase()})")
        io.sleep(TICK_S)
    io.say("저장했습니다")
    return f"{recorded_s:.0f} 초 녹화, 저장, 홈 복귀" + (" (중지로 일찍 끝냄)" if reason else "")
