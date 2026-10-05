"""Glove calibration session: Korean prompts, per-pose retry, no restart on a bad pose.

10.04 on arm4090 every real run ended in "calibration failed" and had to be
restarted from the console: the prompts were English, a pose that moved killed
the whole run, and Enters pressed during a recording recorded the next poses
at once (right hand: open 0.654 vs fist 0.655 rad).
"""

from __future__ import annotations

from motion_acq.console import phases
from motion_acq.hand.calibration import (
    CalibrationError,
    pose_features,
    run_session,
    unsteady_features,
)
from motion_acq.hand.nova2 import features
from motion_acq.hand.retarget import load_hand_retarget_config
from motion_acq.hand.synthetic import POSE_ANGLES

import pytest

CONFIG = load_hand_retarget_config()
PINCHES = ["pinch_index", "pinch_middle", "pinch_ring"]


def still(pose: str, n: int = 20) -> list[dict[str, float]]:
    return [features(POSE_ANGLES[pose], CONFIG.features)] * n


def shaking(pose: str, feature: str, n: int = 20) -> list[dict[str, float]]:
    base = features(POSE_ANGLES[pose], CONFIG.features)
    return [{**base, feature: base[feature] + (0.3 if i % 2 else -0.3)} for i in range(n)]


class Operator:
    """Plays the operator: each pose in turn returns the next scripted take."""

    def __init__(self, takes: dict[str, list[list[dict[str, float]]]]) -> None:
        self.takes = {pose: list(t) for pose, t in takes.items()}
        for pose in CONFIG.poses:  # poses a test does not script (the pinch poses) go well
            self.takes.setdefault(pose, [still(pose)] * 3)
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
    return run_session(side="right", user="t", poses=CONFIG.poses, feature_poses=CONFIG.feature_poses,
                       min_span=CONFIG.min_span_rad, ask=op.ask, record=op.record, say=op.say,
                       relative_min_span=CONFIG.relative_min_span, max_std_relative=CONFIG.pose_std_relative, **kw)


def test_good_takes_need_one_prompt_per_pose():
    op = Operator({p: [still(p)] for p in CONFIG.poses})
    cal = session(op)
    assert set(cal.ranges) == {s.name for s in CONFIG.features}
    assert op.recorded == list(CONFIG.poses)


def test_prompts_are_korean_numbered_and_seen_by_the_console():
    op = Operator({p: [still(p)] for p in CONFIG.poses})
    session(op)
    n = len(CONFIG.poses)
    for i, text in enumerate(op.prompts, 1):
        assert f"[오른손 {i}/{n}]" in text
        assert phases.prompt([], text) == text  # the console shows its Enter button
    assert not any("Hold still" in t or "press Enter" in t for t in op.prompts)
    for pose, text in CONFIG.poses.items():  # what to do, in Korean, in configs/hands/nova2_to_rh56f1.yaml
        assert any("가" <= ch <= "힣" for ch in text), pose


def test_a_pose_that_moved_is_asked_again_not_the_whole_run():
    op = Operator({"open": [still("open")],
                   "fist": [shaking("fist", "index"), still("fist")],
                   "thumb_opposed": [still("thumb_opposed")]})
    session(op)
    assert op.recorded == ["open", "fist", "fist", "thumb_opposed"] + PINCHES
    assert any("움직였습니다" in s for s in op.said)


def test_only_the_features_a_pose_calibrates_must_hold_still():
    # thumb_opposed sets only thumb_opposition; the index finger may drift meanwhile
    assert pose_features(CONFIG.feature_poses, "thumb_opposed") == ("thumb_opposition",)
    assert unsteady_features(shaking("thumb_opposed", "index"), ("thumb_opposition",), 0.05) == {}
    moved = unsteady_features(shaking("thumb_opposed", "thumb_opposition"), ("thumb_opposition",), 0.05)
    assert set(moved) == {"thumb_opposition"}
    op = Operator({"open": [still("open")], "fist": [still("fist")],
                   "thumb_opposed": [shaking("thumb_opposed", "index")]})
    session(op)
    assert op.recorded == list(CONFIG.poses)


def test_indistinct_poses_redo_just_those_poses():
    # 10.04 left: a fist with the thumb left out -> thumb_bend open 0.378 vs fist 0.334
    thumb_out_fist = [{**s, "thumb_bend": still("open")[0]["thumb_bend"]} for s in still("fist")]
    op = Operator({"open": [still("open"), still("open")],
                   "fist": [thumb_out_fist, still("fist")],
                   "thumb_opposed": [still("thumb_opposed")]})
    cal = session(op)
    assert op.recorded == ["open", "fist", "thumb_opposed"] + PINCHES + ["open", "fist"]
    assert any("엄지 굽힘" in s and "거의 같습니다" in s for s in op.said)
    assert cal.ranges["thumb_bend"].closed > cal.ranges["thumb_bend"].open


def test_gives_up_after_the_tries_with_a_korean_reason():
    op = Operator({"open": [shaking("open", "index")] * 3})
    with pytest.raises(CalibrationError, match="3번"):
        session(op, max_tries=3)
    assert op.recorded == ["open"] * 3


def test_each_feature_has_its_own_retries():
    """Review: one shared budget aborted when three different features failed once each."""
    open_take = still("open")
    fist_same_index = [{**s, "index": open_take[0]["index"]} for s in still("fist")]
    fist_same_middle = [{**s, "middle": open_take[0]["middle"]} for s in still("fist")]
    fist_same_ring = [{**s, "ring": open_take[0]["ring"]} for s in still("fist")]
    op = Operator({"open": [still("open")] * 4,
                   "fist": [fist_same_index, fist_same_middle, fist_same_ring, still("fist")],
                   "thumb_opposed": [still("thumb_opposed")]})
    cal = session(op, max_tries=3)
    assert cal.ranges["ring"].closed > cal.ranges["ring"].open
    assert op.recorded.count("fist") == 4


def test_a_glove_dropout_redoes_only_that_pose():
    class Dropout(Operator):
        def record(self, pose):
            if pose == "fist" and self.recorded.count("fist") == 0:
                self.recorded.append(pose)
                raise CalibrationError("2.0 초에 장갑 샘플 3 개뿐")
            return super().record(pose)

    op = Dropout({p: [still(p)] for p in CONFIG.poses})
    session(op)
    assert op.recorded == ["open", "fist", "fist", "thumb_opposed"] + PINCHES
    assert any("샘플 3 개뿐" in s for s in op.said)


def test_pinch_poses_calibrate_tip_distances_relative_to_the_open_hand():
    """10.05: tip distances are in the glove's unit; a touch must be clearly closer than the open hand."""
    op = Operator({p: [still(p)] for p in CONFIG.poses})
    cal = session(op)
    r = cal.ranges["pinch_index"]
    assert r.closed < r.open  # touching = shorter distance
    not_touching = [{**s, "pinch_index": still("open")[0]["pinch_index"] * 0.9} for s in still("pinch_index")]
    op = Operator({"pinch_index": [not_touching, still("pinch_index")], "open": [still("open")] * 2})
    session(op)
    assert any("엄지-검지 집기" in s and "거의 같습니다" in s for s in op.said)


def test_a_glove_without_tip_data_cannot_calibrate_the_pinches():
    no_tips = {p: [[{k: v for k, v in s.items() if not k.startswith("pinch")} for s in still(p)]] * 3
               for p in CONFIG.poses}
    with pytest.raises(CalibrationError, match="tip data"):
        session(Operator(no_tips))
