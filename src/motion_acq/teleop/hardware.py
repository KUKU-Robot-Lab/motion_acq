"""Shared real-teleop preflight checks and required calibration loading."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from motion_acq.calibration.control_tcp import (
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from motion_acq.feetech.calibration import (
    assert_calibrated,
    load_config,
    user_calibration_path,
)
from motion_acq.feetech.setup import list_feetech_serial_ports

log = logging.getLogger("motion_acq.teleop.hardware")


def validate_feetech_ready(args) -> None:
    if args.skip_feetech:
        return
    feetech_config = load_config(args.rig_config)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    assert_calibrated(feetech_config, source=user_calibration_path())
    validate_feetech_ports_exist(feetech_config, robot=args.robot)


def validate_feetech_ports_exist(
    feetech_config,
    *,
    robot: str = "openarmv1",
    list_ports=list_feetech_serial_ports,
) -> None:
    ports = {
        side: getattr(feetech_config, side).port or feetech_config.port
        for side in ("left", "right")
    }
    missing = {
        side: port
        for side, port in ports.items()
        if not port or not Path(port).exists()
    }
    if missing:
        current = sorted(list_ports())
        missing_text = ", ".join(
            f"{side}={port or '<unset>'}" for side, port in missing.items()
        )
        current_text = ", ".join(current) if current else "none"
        raise SystemExit(
            "Feetech port configured in rig.yaml is missing: "
            f"{missing_text}.\n"
            f"Current Feetech ports: {current_text}\n"
            "Remap Feetech without touching CAN/PICO:\n"
            f"  uv run handumi-setup-hardware --robot {robot} --device pico "
            "--skip-can-map --skip-can-repair --skip-pico "
            "--force-feetech-calibration"
        )

    denied = {
        side: port
        for side, port in ports.items()
        if port and not os.access(port, os.R_OK | os.W_OK)
    }
    if denied:
        denied_text = ", ".join(f"{side}={port}" for side, port in denied.items())
        raise SystemExit(
            f"Missing permission to open Feetech: {denied_text}.\n"
            "Run first:\n"
            f"  uv run handumi-setup-hardware --robot {robot} --device pico "
            "--skip-can-map --skip-can-repair --skip-feetech-map --skip-pico"
        )


def load_required_controller_tcp_calibration(args):
    path, source = calibration_path_for_robot_device(
        args.robot,
        args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    if not path.exists():
        raise SystemExit(
            f"Missing controller->TCP calibration: {path}\n"
            "Run the TCP calibration before real teleop, or pass "
            "--controller-tcp-calibration <path>."
        )
    calibration = load_controller_tcp_calibration(path)
    log.info("controller->TCP calibration: %s", source)
    return calibration
