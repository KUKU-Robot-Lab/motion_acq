"""Glove calibration session (example poses, 10.05): Korean prompts, per-pose retry, the
poses the glove cannot tell apart are redone, and a saved calibration is reused.

10.04 on arm4090 every real run ended in "calibration failed" and had to be restarted from
the console; 10.05 the open/fist method moved the wrong fingers and had to be redone every
session (user: that is what the user name is for).
"""

from __future__ import annotations

import pytest
from hand_fixtures import CONFIG, examples, held

from motion_acq.console import phases
from motion_acq.hand.calibration import CalibrationError, run_session, unsteady_inputs

ORDER = list(CONFIG.examples)


def shaking(pose: str, signal: str) -> list[dict[str, float]]:
    return [{**s, signal: s[signal] + (0.3 if i % 2 else -0.3)} for i, s in enumerate(held(pose))]


class Operator:
    """Plays the operator: each pose in turn returns the next scripted take (default: held well)."""

    def __init__(self, takes: dict[str, list[list[dict[str, float]]]] | None = None) -> None:
        self.takes = {pose: list(t) for pose, t in (takes or {}).items()}
        for pose in ORDER:
            self.takes.setdefault(pose, [held(pose, seed=k) for k in range(3)])
        self.prompts: list[str] = []
        self.said: list[str] = []
        self.recorded: list[str] = []

    def ask(self, text: str) -> None:
        self.prompts.append(text)

    def record(self, pose: str) -> list[dict[str, float]]:
        self.recorded.append(pose)
        return self.takes[pose].pop(0)

    def say(self, text: str) -> None:
        self.said.append(text)


def session(op: Operator, **kw):
    return run_session(side="right", user="t", groups=CONFIG.groups, examples=examples(),
                       ask=op.ask, record=op.record, say=op.say, **kw)


def test_good_takes_need_one_prompt_per_pose():
    op = Operator()
    cal = session(op)
    assert op.recorded == ORDER
    assert set(cal.medians) == set(ORDER)


def test_prompts_are_korean_numbered_and_seen_by_the_console():
    op = Operator()
    session(op)
    n = len(ORDER)
    for i, text in enumerate(op.prompts, 1):
        assert f"[오른손 {i}/{n}]" in text
        assert phases.prompt([], text) == text  # the console shows its Enter button
    for pose, example in CONFIG.examples.items():
        assert any("가" <= ch <= "힣" for ch in example.prompt), pose


def test_a_pose_that_moved_is_asked_again_not_the_whole_run():
    op = Operator({"fist": [shaking("fist", "index_pip"), held("fist")]})
    session(op)
    assert op.recorded == ORDER[:3] + ["fist"] + ORDER[3:]
    assert any("움직였습니다" in s for s in op.said)


def test_tip_distances_may_wobble_a_little():
    takes = held("pinch_index")
    wobbly = [{**s, "tipdist_index": s["tipdist_index"] * (1.05 if i % 2 else 0.95)} for i, s in enumerate(takes)]
    assert unsteady_inputs(wobbly, ("tipdist_index",)) == {}
    assert unsteady_inputs(shaking("open", "tipdist_index"), ("tipdist_index",)) == {}  # 0.3 mm of ~100
    assert set(unsteady_inputs(shaking("open", "index_pip"), ("index_pip",))) == {"index_pip"}


def test_two_poses_the_glove_cannot_tell_apart_are_redone():
    """E.g. the 'index only' pose done with the hand still flat: same glove reading, different robot pose."""
    op = Operator({"index": [held("flat", seed=7), held("index")]})
    session(op)
    assert op.recorded[:len(ORDER)] == ORDER
    assert op.recorded[len(ORDER):] == ["index"]  # the one pose that collides with the others
    assert any("비슷하게 읽히는" in s for s in op.said)


def test_gives_up_after_the_tries_with_a_korean_reason():
    op = Operator({"open": [shaking("open", "index_pip")] * 3})
    with pytest.raises(CalibrationError, match="3번"):
        session(op, max_tries=3)
    assert op.recorded == ["open"] * 3


def test_a_glove_dropout_redoes_only_that_pose():
    class Dropout(Operator):
        def record(self, pose):
            if pose == "fist" and self.recorded.count("fist") == 0:
                self.recorded.append(pose)
                raise CalibrationError("2.0 초에 장갑 샘플 3 개뿐")
            return super().record(pose)

    op = Dropout()
    session(op)
    assert op.recorded == ORDER[:3] + ["fist"] + ORDER[3:]
    assert any("샘플 3 개뿐" in s for s in op.said)


def test_a_glove_without_tip_data_cannot_calibrate():
    no_tips = {p: [[{k: v for k, v in s.items() if not k.startswith("tipdist")} for s in held(p)]] * 3
               for p in ORDER}
    with pytest.raises(CalibrationError, match="tip data"):
        session(Operator(no_tips))


def test_the_fewest_poses_are_redone():
    from motion_acq.hand.calibration import poses_to_redo

    assert poses_to_redo([("flat", "index"), ("index", "middle"), ("index", "ring")]) == ["index"]
    assert sorted(poses_to_redo([("a", "b"), ("c", "d")])) == ["a", "c"]
