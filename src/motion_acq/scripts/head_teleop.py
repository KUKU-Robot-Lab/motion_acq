#!/usr/bin/env python3
"""Meta Quest HMD -> Dynamixel pan/tilt head (STEP 2), separate from the arms.

    macq head                          # fake head bus (default), Quest via rig
    macq head --axes pan               # staged test: tilt stays at its anchor
    macq head --backend real           # opens the station head port

Start and end pose is home (head.pan/tilt.home_deg): after torque on the head
walks there at head.limits.home_speed_deg_s (default 20 deg/s), and walks back
on Ctrl+C, SIGTERM, --duration-s or a bus fault (not on a hardware alert,
which stops in place). The head anchors after the HMD has been tracked for
--auto-start-delay-s; it then follows yaw (pan) and pitch (tilt) relative to
that moment, around home. HMD loss holds the head. After the return the head
holds home (torque on) unless --torque-off-on-exit is given.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np

from motion_acq.calibration.control_tcp import ControllerTcpCalibration
from motion_acq.config import DEFAULT_RIG_CONFIG, STATION_ENV
from motion_acq.cpu import keep_off_rt
from motion_acq.head.config import HeadConfig, load_head_config
from motion_acq.head.dynamixel import (
    FakeHeadBus,
    HeadAlertError,
    HeadBus,
    HeadBusError,
    HeadDriver,
    SdkHeadBus,
    deg_to_tick,
)
from motion_acq.head.retarget import HeadRetargeter
from motion_acq.head.session import HeadSession, open_log
from motion_acq.robots.utils import IDENTITY_POSE7
from motion_acq.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider

log = logging.getLogger("motion_acq.head")
STATUS_PERIOD_S = 1.0
MAX_RATE_HZ = 200.0


def _positive_rate(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= MAX_RATE_HZ:
        raise argparse.ArgumentTypeError(f"rate must be in (0, {MAX_RATE_HZ:g}] Hz")
    return value


def _sigterm_to_interrupt(signum, frame) -> None:  # noqa: ARG001
    # kill <pid> (SIGTERM) is the standard stop here: run the same cleanup as Ctrl+C.
    raise KeyboardInterrupt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    p.add_argument("--backend", choices=("fake", "real"), default="fake")
    p.add_argument("--axes", choices=("pan", "tilt", "both"), default="both")
    p.add_argument("--quest-ip", default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument("--auto-start-delay-s", type=float, default=2.0)
    p.add_argument("--rate-hz", type=_positive_rate, default=None, help="Override head.rate_hz.")
    p.add_argument("--duration-s", type=float, default=0.0, help="Stop after N s (0 = until Ctrl+C).")
    p.add_argument("--log-dir", type=Path, default=Path("logs/head"))
    p.add_argument("--no-log", action="store_true")
    p.add_argument(
        "--udp-target", default=None, metavar="HOST:PORT",
        help="Also send each cycle to the recorder (teleop-record --sidecar head=PORT).",
    )
    p.add_argument("--torque-off-on-exit", action="store_true")
    return p.parse_args(argv)


def _udp_target(text: str | None) -> tuple[str, int] | None:
    if not text:
        return None
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise SystemExit(f"--udp-target expects HOST:PORT, not {text!r}")
    return host, int(port)


def build_tracker(args: argparse.Namespace) -> MetaQuestTrackingProvider:
    base = MetaQuestConfig.from_yaml(args.rig_config)
    config = MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else base.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else base.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else base.sync_port,
        connect_retry_s=base.connect_retry_s,
        frame_stale_timeout_s=base.frame_stale_timeout_s,
    )
    identity = IDENTITY_POSE7.astype(np.float32)
    calibration = ControllerTcpCalibration(left=identity.copy(), right=identity.copy(), source=None)
    return MetaQuestTrackingProvider(config=config, calibration=calibration, reset_workspace_on_x=False)


def build_bus(args: argparse.Namespace, config: HeadConfig) -> HeadBus:
    if args.backend == "real":
        if not config.from_station or not config.port:
            raise SystemExit("--backend real needs a head section with a port in the station rig.")
        return SdkHeadBus(config.port, config.baud)
    hw, retarget = config.hardware, config.retarget
    return FakeHeadBus(
        model=hw.model_number,
        present_ticks={
            hw.pan_id: deg_to_tick(retarget.pan.home_deg),
            hw.tilt_id: deg_to_tick(retarget.tilt.home_deg),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    log.info(keep_off_rt())  # off the RH56F1 EtherCAT cores (sim2real cpu plan)
    config = load_head_config(args.rig_config, allow_fake_default=args.backend == "fake")
    if not config.from_station:
        log.warning("No head section in %s; fake head uses home 0/0.", args.rig_config)
    rate_hz = args.rate_hz if args.rate_hz is not None else config.rate_hz
    if not 0.0 < rate_hz <= MAX_RATE_HZ:
        raise SystemExit(f"head.rate_hz {rate_hz} is outside (0, {MAX_RATE_HZ:g}].")
    retarget = config.retarget.with_axes(args.axes)
    log.info(
        "Head %s backend, axes=%s, pan window [%.1f, %.1f], tilt window [%.1f, %.1f] deg, %.0f Hz.",
        args.backend, args.axes, *config.pan_window, *config.tilt_window, rate_hz,
    )
    driver = HeadDriver(
        build_bus(args, config), config.hardware,
        pan_window=config.pan_window, tilt_window=config.tilt_window,
    )
    tracker = build_tracker(args)
    log_file = None if args.no_log else open_log(args.log_dir, os.environ.get(STATION_ENV, ""))
    signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    tracker.start()
    alert = False
    try:
        measured = driver.start()
        log.info("Head torque on at pan %.1f / tilt %.1f deg; moving to home %.1f / %.1f at %.0f deg/s.",
                 *measured, *config.home, config.home_speed_deg_s)
        measured = driver.move_to(*config.home, speed_deg_s=config.home_speed_deg_s, rate_hz=rate_hz)
        log.info("Head at home (pan %.1f / tilt %.1f deg); following the HMD after anchoring.", *measured)
        session = HeadSession(
            tracker, driver, HeadRetargeter(retarget),
            auto_start_delay_s=args.auto_start_delay_s,
            max_consecutive_faults=config.max_consecutive_faults,
            log_file=log_file,
            udp_target=_udp_target(args.udp_target),
        )
        _run(session, rate_hz, args.duration_s)
    except HeadBusError as exc:
        alert = isinstance(exc, HeadAlertError)
        raise SystemExit(f"Head stopped: {exc}") from exc
    except KeyboardInterrupt:
        log.info("Stopping head.")
    finally:
        try:
            tracker.stop()
        except Exception:  # noqa: BLE001 - never skip the driver cleanup
            log.exception("tracker.stop() failed")
        try:
            _return_home(driver, config, rate_hz, alert=alert)
            driver.stop(torque_off=args.torque_off_on_exit)
        finally:
            if log_file is not None:
                log_file.close()


def _return_home(driver: HeadDriver, config: HeadConfig, rate_hz: float, *, alert: bool) -> None:
    """End pose = home, like the start pose. A hardware alert stops in place."""
    if not driver.started:
        return
    if alert:
        log.error("Hardware alert: the head stays where it is (not returning home).")
        return
    try:
        measured = driver.move_to(*config.home, speed_deg_s=config.home_speed_deg_s, rate_hz=rate_hz)
        log.info("Head back at home (pan %.1f / tilt %.1f deg).", *measured)
    except (HeadBusError, KeyboardInterrupt) as exc:
        log.error("Head could not return home: %s", exc or "interrupted")


def _run(session: HeadSession, rate_hz: float, duration_s: float) -> None:
    period = 1.0 / rate_hz
    start = last_status = time.monotonic()
    while duration_s <= 0.0 or time.monotonic() - start < duration_s:
        cycle = time.monotonic()
        step = session.tick()
        if cycle - last_status >= STATUS_PERIOD_S:
            last_status = cycle
            _status(session, step)
        time.sleep(max(period - (time.monotonic() - cycle), 0.0))
    s = session.stats
    log.info("Head done: %d cycles, %d commands, %d hold cycles, %d re-anchors, %d faults.",
             s.cycles, s.commands, s.holds, s.reanchors, s.faults)


def _status(session: HeadSession, step) -> None:
    def fmt(value: float | None) -> str:
        return "  -  " if value is None else f"{value:+6.1f}"

    meas = session.measured or (None, None)
    state = step.state.value if step else session.retargeter.state.value
    rel = (step.rel_yaw_deg, step.rel_pitch_deg) if step else (None, None)
    cmd = (step.command_pan_deg, step.command_tilt_deg) if step else (None, None)
    log.info("head %-7s rel yaw %s pitch %s | cmd pan %s tilt %s | meas pan %s tilt %s",
             state, fmt(rel[0]), fmt(rel[1]), fmt(cmd[0]), fmt(cmd[1]), fmt(meas[0]), fmt(meas[1]))


if __name__ == "__main__":
    main()
