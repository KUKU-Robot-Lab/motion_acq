"""Stored rest (all joints 0, s2r "차렷") <-> home arm paths, as sim2real uses them.

sim2real moves the RH56F1 arms between rest and the rh_aglt home only along
paths planned offline (RRT, >= 2 cm clearance to the table and the body, max
0.3 rad/s, hand closed as a fist) and forbids an unverified straight joint
line between the two (docs RUNBOOK_LEFT_REAL_2026-08-31: "검증 안 된 관절
직선 ... 금지"). motion_acq uses the same files (configs/paths/, copied from
sim2real deploy/policy_control/paths/, see configs/paths/README.md):

- start: an arm at rest plays the path forward to home; an arm already at
  home only settles; any other pose is refused (bring it to rest or home with
  the sim2real home step first);
- end: from home, the path is played backwards to rest before the motors
  are disabled, so the arm is not dropped from the home pose.

The paths assume the RH56F1 is closed (fist) while the arm moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_DOF = 7
PATH_START_TOLERANCE_RAD = 0.05  # sim2real check_path_start
# An unpowered arm hanging at rest drifts a few degrees on the wrist (10.04 right:
# j5 6.4, j6 -4.5, j7 4.0 deg). Within this of rest it is first brought to exact
# rest slowly, then the path starts; beyond it the start is refused.
REST_ALIGN_MAX_RAD = 0.15
REST_ALIGN_SPEED_RAD_S = 0.05
MAX_PATH_SPEED_RAD_S = 0.35  # refuse files faster than the planned 0.3 rad/s (+ margin)
_PREFIX = {"right": "r_aj_", "left": "l_aj_"}


class HomePathError(ValueError):
    pass


@dataclass(frozen=True)
class HomePath:
    side: str
    q: np.ndarray  # (N, 7): rest -> home
    dt: float

    @property
    def rest(self) -> np.ndarray:
        return self.q[0]

    @property
    def home(self) -> np.ndarray:
        return self.q[-1]

    @property
    def duration_s(self) -> float:
        return (len(self.q) - 1) * self.dt


def load_home_path(path: Path, side: str) -> HomePath:
    try:
        data = np.load(Path(path), allow_pickle=False)
        q = np.asarray(data["arm_target"], dtype=np.float64)
        dt = float(data["meta_step_dt"])
        joints = [str(j) for j in data["meta_joints"]]
    except (OSError, KeyError, ValueError) as exc:
        raise HomePathError(f"{path}: not a sim2real home path ({exc})") from exc
    expected = [f"{_PREFIX[side]}{i}" for i in range(1, ARM_DOF + 1)]
    if joints != expected:
        raise HomePathError(f"{path}: joints {joints}, expected {expected} for the {side} arm")
    if q.ndim != 2 or q.shape[1] != ARM_DOF or len(q) < 2 or not np.all(np.isfinite(q)) or not dt > 0:
        raise HomePathError(f"{path}: arm_target {q.shape} / step {dt} is not a usable path")
    speed = float(np.abs(np.diff(q, axis=0)).max() / dt)
    if speed > MAX_PATH_SPEED_RAD_S:
        raise HomePathError(f"{path}: {speed:.2f} rad/s step, above {MAX_PATH_SPEED_RAD_S} rad/s")
    if np.abs(q[0]).max() > 1e-6:
        raise HomePathError(f"{path}: does not start at rest (all joints 0)")
    return HomePath(side=side, q=q, dt=dt)


def classify_start(measured: np.ndarray, path: HomePath, home: np.ndarray, home_tolerance_rad: float) -> str:
    """'rest' (play the path), 'near_rest' (align slowly, then the path), 'home' (only settle)
    or raise if neither."""
    measured = np.asarray(measured, dtype=np.float64)
    if not np.allclose(path.home, home, atol=1e-3):
        raise HomePathError(f"{path.side} path ends at {np.round(path.home, 4).tolist()}, "
                            f"not at the home pose {np.round(home, 4).tolist()}; replan it in sim2real")
    off_rest = float(np.abs(measured - path.rest).max())
    if off_rest <= PATH_START_TOLERANCE_RAD:
        return "rest"
    if off_rest <= REST_ALIGN_MAX_RAD:
        return "near_rest"
    if np.abs(measured - home).max() <= home_tolerance_rad:
        return "home"
    raise HomePathError(
        f"OpenArm {side_label(path.side)} is at {np.round(np.rad2deg(measured), 1).tolist()} deg: neither "
        f"rest (all 0, within {np.rad2deg(REST_ALIGN_MAX_RAD):.1f} deg) nor home. Bring it to rest "
        "or home with the sim2real home step first; motion_acq only moves along the stored path."
    )


def side_label(side: str) -> str:
    return f"{side} arm"
