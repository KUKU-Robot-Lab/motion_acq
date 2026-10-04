#!/usr/bin/env python3
"""Which way does the head camera turn? Small real head moves + the head camera image.

    MACQ_STATION=arm4090 .venv/bin/python scripts/head_direction_check.py [--step-deg 5] [--camera 4]

Moves the real head (approval first): home, then pan home+step, back, tilt
home+step, back, at 10 deg/s, and grabs a camera frame at each pose. The
image shift between home and each move (median ORB feature motion) says where the
camera looked:
    scene moves right in the image -> camera turned left
    scene moves up in the image    -> camera looked down
and from that the head.pan/tilt.sign that makes HMD yaw left -> camera left
and HMD pitch up -> camera up (the retargeter: HMD left/up = positive
command). Nothing else moves; the head ends at home, torque on.
logs/head/dir_<axis>_pair.jpg (home | moved) is kept for a look by eye.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion_acq.config import DEFAULT_RIG_CONFIG  # noqa: E402
from motion_acq.head.config import load_head_config  # noqa: E402
from motion_acq.head.dynamixel import HeadDriver, SdkHeadBus  # noqa: E402

MIN_SHIFT_PX = 3.0  # a 5 deg turn moves a 640 px / ~69 deg image by ~45 px
MIN_MATCHES = 20


def image_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """(dx, dy) in pixels of the scene from before to after (+x right, +y down).

    Median displacement of ORB keypoint matches on CLAHE-equalised grey images:
    the RealSense auto exposure changes brightness between the two frames, which
    made phase correlation report the wrong direction (10.04 pan).
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grey = [clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in (before, after)]
    orb = cv2.ORB_create(nfeatures=2000)
    (kp_a, des_a), (kp_b, des_b) = (orb.detectAndCompute(g, None) for g in grey)
    if des_a is None or des_b is None:
        raise ValueError("no features in one of the frames")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b)
    if len(matches) < MIN_MATCHES:
        raise ValueError(f"only {len(matches)} feature matches; scene too plain or moved too far")
    shifts = np.array([np.subtract(kp_b[m.trainIdx].pt, kp_a[m.queryIdx].pt) for m in matches])
    dx, dy = np.median(shifts, axis=0)
    return float(dx), float(dy)


def sign_from_shift(axis: str, step_deg: float, shift: tuple[float, float]) -> tuple[int | None, str]:
    """head.<axis>.sign so that a positive command (HMD left/up) turns the camera left/up."""
    dx, dy = shift
    if axis == "pan":
        if abs(dx) < MIN_SHIFT_PX:
            return None, f"pan +{step_deg:g} deg: no clear image shift ({dx:+.1f} px)"
        camera_left = dx > 0  # scene moved right
        return (1 if camera_left else -1), f"pan +{step_deg:g} deg: scene {dx:+.1f} px -> camera turned " \
            f"{'left' if camera_left else 'right'}"
    if abs(dy) < MIN_SHIFT_PX:
        return None, f"tilt +{step_deg:g} deg: no clear image shift ({dy:+.1f} px)"
    camera_up = dy > 0  # scene moved down
    return (1 if camera_up else -1), f"tilt +{step_deg:g} deg: scene {dy:+.1f} px -> camera looked " \
        f"{'up' if camera_up else 'down'}"


def grab(cap, settle_s: float = 0.6) -> np.ndarray:
    end = time.monotonic() + settle_s
    frame = None
    while time.monotonic() < end or frame is None:  # drain buffered frames after the move
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera read failed")
    return frame


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    ap.add_argument("--camera", type=int, default=4, help="V4L2 index of the head RealSense color stream")
    ap.add_argument("--step-deg", type=float, default=5.0)
    ap.add_argument("--save-dir", type=Path, default=ROOT / "logs" / "head")
    args = ap.parse_args()
    if not 1.0 <= args.step_deg <= 10.0:
        raise SystemExit("--step-deg must be within 1..10")
    config = load_head_config(args.rig_config, allow_fake_default=False)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    driver = HeadDriver(SdkHeadBus(config.port, config.baud), config.hardware,
                        pan_window=config.pan_window, tilt_window=config.tilt_window)
    pan0, tilt0 = config.home
    speed = 10.0
    tol = config.home_tolerance_deg
    results = {}
    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"head torque on at {driver.start()}; to home ({pan0}, {tilt0})")
        driver.move_to(pan0, tilt0, speed_deg_s=speed, tolerance_deg=tol)
        home = grab(cap)
        cv2.imwrite(str(args.save_dir / "dir_home.jpg"), home)
        for axis, target in (("pan", (pan0 + args.step_deg, tilt0)), ("tilt", (pan0, tilt0 + args.step_deg))):
            print(f"{axis}: to {target}")
            measured = driver.move_to(*target, speed_deg_s=speed, tolerance_deg=tol)
            moved = grab(cap)
            cv2.imwrite(str(args.save_dir / f"dir_{axis}.jpg"), moved)
            cv2.imwrite(str(args.save_dir / f"dir_{axis}_pair.jpg"), np.hstack([home, moved]))  # home | moved
            results[axis] = (measured, image_shift(home, moved))
            driver.move_to(pan0, tilt0, speed_deg_s=speed, tolerance_deg=tol)
    finally:
        try:
            if driver.started:
                driver.move_to(pan0, tilt0, speed_deg_s=speed, tolerance_deg=tol)
        finally:
            driver.stop()
            cap.release()
    print()
    for axis, (measured, shift) in results.items():
        sign, text = sign_from_shift(axis, args.step_deg, shift)
        current = getattr(config.retarget, axis).sign
        verdict = "?" if sign is None else ("matches the config" if sign == current else "CHANGE the config")
        print(f"{text}; measured {measured}. head.{axis}.sign should be {sign} (config {current:+g}): {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
