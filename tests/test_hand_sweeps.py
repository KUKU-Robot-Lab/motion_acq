"""[손가락별 맞춤] sweeps: per-finger 0..1 ranges and glove crosstalk removal (10.11 user)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hand_fixtures import CONFIG, make_calibration

from motion_acq.hand.calibration import HandCalibration, rezero
from motion_acq.hand.flexion import FLEX_INPUTS, FlexionRetargeter
from motion_acq.hand.sweeps import SWEEPS, FingerSweeps, SweepError, fit_sweeps, report, unsteady
from motion_acq.hand.synthetic import POSE_ANGLES, synthetic_angles

CLOSED = {j: CONFIG.limits_rad[j][1] for j in FLEX_INPUTS}
THUMB_TO_INDEX = 0.6  # leak of the thumb bend (thumb_pip) into the index reading
MIDDLE_TO_INDEX = 0.3  # leak of the middle bend into the index reading


def leaky(curl: dict[str, float], bend: float = 0.0, oppose: float = 0.0) -> dict[str, float]:
    """A glove whose index reading moves with the thumb bend and the middle finger (as 10.11 logs)."""
    a = synthetic_angles(curl, bend, oppose)
    middle = (a["middle_mcp"] + a["middle_pip"]) - (POSE_ANGLES["flat"]["middle_mcp"] + POSE_ANGLES["flat"]["middle_pip"])
    a["index_pip"] += THUMB_TO_INDEX * (a["thumb_pip"] - 0.01) + MIDDLE_TO_INDEX * middle
    return a


def sweep(kind: str, seed: int = 0, cycles: int = 3, n: int = 240) -> tuple[list, list]:
    rng = np.random.default_rng(seed)

    def noisy(a):
        return {k: v + float(rng.normal(0.0, 0.004)) for k, v in a.items()}

    rest = [noisy(leaky({})) for _ in range(30)]
    moving = []
    for i in range(n):
        u = 0.5 - 0.5 * math.cos(2 * math.pi * cycles * i / n)  # 0 -> 1 -> 0, `cycles` times
        if kind == "thumb":
            v = 0.5 - 0.5 * math.cos(2 * math.pi * (cycles + 1) * i / n)
            moving.append(noisy(leaky({}, bend=u, oppose=0.6 * v)))
        elif kind == "fist":
            moving.append(noisy(leaky({f: u for f in ("index", "middle", "ring")}, bend=u)))
        else:
            moving.append(noisy(leaky({kind: u})))
    return rest, moving


@pytest.fixture(scope="module")
def sweeps() -> FingerSweeps:
    return fit_sweeps({k: sweep(k, i) for i, k in enumerate(SWEEPS)}, recorded="test")


def run(fx: FlexionRetargeter, signals, n: int = 5) -> dict[str, float]:
    out = {}
    for _ in range(n):
        out = fx.solve(signals)
    return out


def test_fit_finds_the_crosstalk_and_nothing_else(sweeps):
    thumb = sweeps.leak["index_1"]["thumb"]
    assert thumb[0] == pytest.approx(THUMB_TO_INDEX, abs=0.1)  # thumb_pip coefficient
    assert sweeps.leak["index_1"]["middle"][0] == pytest.approx(MIDDLE_TO_INDEX, abs=0.05)
    assert sweeps.leak["middle_1"] == {} and "index" not in sweeps.leak["ring_1"]
    lines = "\n".join(report(sweeps))
    assert "엄지 → 검지" in lines and "중지 → 검지" in lines


def test_thumb_alone_no_longer_moves_the_robot_index(sweeps):
    medians = make_calibration("right").medians
    plain = FlexionRetargeter(medians, CLOSED, CONFIG.flexion)
    fixed = FlexionRetargeter(medians, CLOSED, CONFIG.flexion, sweeps)
    thumb = leaky({}, bend=1.0, oppose=0.3)
    assert run(plain, thumb)["index_1"] > 0.1  # the problem: the glove's index reading moved
    q = run(fixed, thumb)
    assert q["index_1"] < 0.05 and q["thumb_2"] > 0.8 * CLOSED["thumb_2"]
    q = run(fixed, leaky({"middle": 1.0}))
    assert q["index_1"] < 0.05 and q["middle_1"] == pytest.approx(CLOSED["middle_1"], abs=0.02)


@pytest.mark.parametrize("finger", ["index", "middle", "ring"])
def test_each_finger_alone_goes_zero_to_one(sweeps, finger):
    fx = FlexionRetargeter({}, CLOSED, CONFIG.flexion, sweeps)
    joint = f"{finger}_1"
    assert run(fx, leaky({}))[joint] == 0.0
    half = run(fx, leaky({finger: 0.5}))[joint]
    assert 0.3 * CLOSED[joint] < half < 0.7 * CLOSED[joint]
    q = run(fx, leaky({finger: 1.0}))
    assert q[joint] == pytest.approx(CLOSED[joint], abs=0.05)
    assert all(q[j] < 0.05 for j in ("index_1", "middle_1", "ring_1") if j != joint and not (
        finger == "ring" and j == "pinky_1"))


def test_fist_closes_every_finger(sweeps):
    q = run(FlexionRetargeter({}, CLOSED, CONFIG.flexion, sweeps),
            leaky({f: 1.0 for f in ("index", "middle", "ring")}, bend=1.0))
    assert q == pytest.approx(CLOSED, abs=0.05)


def test_reference_take_moves_the_crosstalk_baseline(sweeps):
    fx = FlexionRetargeter({}, CLOSED, CONFIG.flexion, sweeps)
    c_shape = leaky({}, bend=0.3, oppose=0.4)  # the robot home's C: thumb a little bent and across
    fx.take_reference([c_shape] * 20)
    assert fx.rest["thumb"] != sweeps.rest["thumb"]
    assert run(fx, c_shape)["index_1"] == 0.0
    assert run(fx, leaky({}, bend=1.0, oppose=0.4))["index_1"] < 0.05
    fx.reset()
    assert fx.rest == dict(sweeps.rest)


def test_saved_and_rezeroed_calibrations_keep_the_sweeps(tmp_path, sweeps):
    cal = make_calibration("right").with_sweeps(sweeps)
    path = tmp_path / "op_right.yaml"
    cal.save(path)
    loaded = HandCalibration.load(path, side="right")
    assert loaded.sweeps == sweeps
    assert rezero(loaded, [POSE_ANGLES["open"]] * 10).sweeps == sweeps
    assert HandCalibration.load(path).sweeps is not None
    raw = sweeps.to_dict()
    del raw["closed"]["index_1"]
    with pytest.raises(SweepError):
        FingerSweeps.from_dict(raw)


def test_a_sweep_that_did_not_move_or_a_restless_open_hand_is_refused():
    recs = {k: sweep(k, i) for i, k in enumerate(SWEEPS)}
    rest, _ = recs["index"]
    recs["index"] = (rest, [leaky({"index": 0.1})] * 100)
    with pytest.raises(SweepError, match="검지"):
        fit_sweeps(recs)
    assert unsteady(sweep("index")[0]) == {}
    assert "index_pip" in unsteady([leaky({"index": u / 10}) for u in range(10)])
