#!/usr/bin/env python3
"""Where does the head tilt stop going up? Slow real sweep (approval first).

    MACQ_STATION=arm4090 .venv/bin/python scripts/head_tilt_limit_probe.py [--stop-deg -5]

Moves only the tilt: home, then up (smaller encoder angle) at --speed deg/s
toward --stop-deg, reading the present position and current every tick. It
stops at the first sign of contact:
    the measured angle lags the goal by more than --max-lag-deg, or
    |current| rises more than --current-margin-ma above the home baseline, or
    a hardware error is latched,
then backs off 3 deg and walks home at 10 deg/s (also on Ctrl+C). The free
upper end (contact - margin) goes into head.tilt.range_pos_deg by hand.
logs/head/tilt_probe_<time>.csv keeps every tick.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion_acq.config import DEFAULT_RIG_CONFIG  # noqa: E402
from motion_acq.head.config import load_head_config  # noqa: E402
from motion_acq.head.dynamixel import (  # noqa: E402
    ADDR_PRESENT_POSITION,
    HeadDriver,
    SdkHeadBus,
    tick_to_deg,
)

ADDR_PRESENT_CURRENT = 126  # XC330: 2 bytes, signed, 1 mA
RATE_HZ = 50.0
BACK_OFF_DEG = 3.0
HOME_SPEED_DEG_S = 10.0
BASELINE_S = 1.0


def signed16(raw: int) -> int:
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


IIO_ROOT = Path("/sys/bus/iio/devices")


def find_accel(root: Path = IIO_ROOT) -> Path | None:
    """The head D435i accelerometer (kernel HID sensor, readable without sudo), or None."""
    for dev in sorted(root.glob("iio:device*")):
        name = dev / "name"
        if name.is_file() and name.read_text().strip() == "accel_3d":
            return dev
    return None


def camera_pitch_deg(x: float, y: float, z: float) -> float:
    """Optical-axis elevation from the raw D435i HID accel (negative = looks down).

    The raw z axis points out of the back of the camera: at the arm4090 home it
    reads -72.1 deg, against -72.8 from the depth table plane (10.12)."""
    return -math.degrees(math.atan2(z, math.hypot(x, y)))


def read_pitch(dev: Path | None) -> float | None:
    if dev is None:
        return None
    try:
        x, y, z = (int((dev / f"in_accel_{a}_raw").read_text()) for a in "xyz")
    except (OSError, ValueError):
        return None
    return camera_pitch_deg(x, y, z)


def contact(goal: float, present: float, current_ma: float, baseline_ma: float, *,
            max_lag_deg: float, current_margin_ma: float) -> str | None:
    """Why the sweep must stop now, or None."""
    if abs(goal - present) > max_lag_deg:
        return f"lag {goal - present:+.1f} deg (goal {goal:.1f}, at {present:.1f})"
    if abs(current_ma) > abs(baseline_ma) + current_margin_ma:
        return f"current {current_ma:.0f} mA (home {baseline_ma:.0f} mA)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stop-deg", type=float, default=-5.0, help="highest tilt goal (deg, smaller = up)")
    ap.add_argument("--speed", type=float, default=5.0, help="sweep speed (deg/s)")
    ap.add_argument("--max-lag-deg", type=float, default=3.0)
    ap.add_argument("--current-margin-ma", type=float, default=150.0)
    args = ap.parse_args()

    cfg = load_head_config(DEFAULT_RIG_CONFIG, allow_fake_default=False)
    home_pan, home_tilt = cfg.retarget.pan.home_deg, cfg.retarget.tilt.home_deg
    if not args.stop_deg < home_tilt:
        raise SystemExit(f"--stop-deg {args.stop_deg} must be above home ({home_tilt}): smaller angle")
    pan_window = cfg.pan_window
    tilt_window = (args.stop_deg, cfg.tilt_window[1])
    driver = HeadDriver(SdkHeadBus(cfg.port, cfg.baud), cfg.hardware,
                        pan_window=pan_window, tilt_window=tilt_window)
    tilt_id = cfg.hardware.tilt_id
    out = ROOT / "logs" / "head" / f"tilt_probe_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    def current_ma() -> float:
        return float(signed16(driver.bus.read4(tilt_id, ADDR_PRESENT_CURRENT)))

    def present() -> float:
        return tick_to_deg(driver.bus.read4(tilt_id, ADDR_PRESENT_POSITION))

    accel = find_accel()
    print(f"camera IMU: {accel or 'not found (pitch column stays empty)'}", flush=True)
    driver.start()
    reason = "reached --stop-deg"
    limit = None
    try:
        driver.move_to(home_pan, home_tilt, speed_deg_s=HOME_SPEED_DEG_S, tolerance_deg=cfg.home_tolerance_deg)
        samples = []
        t_end = time.monotonic() + BASELINE_S
        while time.monotonic() < t_end:
            samples.append(current_ma())
            time.sleep(1.0 / RATE_HZ)
        baseline = sorted(samples)[len(samples) // 2]
        print(f"home tilt {present():.2f} deg, current {baseline:.0f} mA; sweeping up to {args.stop_deg} "
              f"at {args.speed} deg/s", flush=True)
        goal = home_tilt
        step = args.speed / RATE_HZ
        t0 = time.monotonic()
        with out.open("w", newline="") as f:
            log = csv.writer(f)
            log.writerow(["t_s", "goal_deg", "present_deg", "current_ma", "camera_pitch_deg"])
            last_print = goal
            while True:
                goal = max(args.stop_deg, goal - step)
                driver.command(home_pan, goal)
                time.sleep(1.0 / RATE_HZ)
                at, amps, pitch = present(), current_ma(), read_pitch(accel)
                pitch_s = "" if pitch is None else f"{pitch:.2f}"
                log.writerow([f"{time.monotonic() - t0:.3f}", f"{goal:.2f}", f"{at:.2f}", f"{amps:.0f}", pitch_s])
                if last_print - goal >= 5.0:
                    print(f"  goal {goal:6.1f}  at {at:6.1f}  current {amps:5.0f} mA  camera {pitch_s or '-'} deg",
                          flush=True)
                    last_print = goal
                why = contact(goal, at, amps, baseline, max_lag_deg=args.max_lag_deg,
                              current_margin_ma=args.current_margin_ma)
                errors = driver.hardware_errors()
                if errors:
                    why = f"hardware error {errors}"
                if why:
                    reason, limit = why, at
                    driver.command(home_pan, at + BACK_OFF_DEG)  # back off downward first
                    time.sleep(0.5)
                    break
                if goal <= args.stop_deg and abs(at - goal) < 1.0:
                    limit = at
                    break
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    finally:
        try:
            driver.move_to(home_pan, home_tilt, speed_deg_s=HOME_SPEED_DEG_S,
                            tolerance_deg=cfg.home_tolerance_deg, settle_timeout_s=10.0)
        finally:
            driver.stop()
    print(f"stopped: {reason}; highest tilt reached {limit if limit is None else round(limit, 1)} deg; "
          f"head back at home. Log: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
