"""Shared command-line configuration for physical-robot teleoperation."""

from __future__ import annotations

import argparse
from pathlib import Path

from motion_acq.config import DEFAULT_RIG_CONFIG
from motion_acq.real.registry import REAL_BACKEND_NAMES
from motion_acq.teleop.common import SIDE_CHOICES
from motion_acq.teleop.motion import add_teleop_motion_arguments
from motion_acq.teleop.standby import GRIPPER_PARK_HOLD_S

DEFAULT_TRANSLATION_SCALE = 1.7
DEFAULT_TRACKING_STALE_MS = 150.0
DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S = 30.0


def _camera_list(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one camera name is required")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("camera names must be unique")
    return names


def add_physical_teleop_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the controls shared by live and recording real-robot teleop."""
    parser.add_argument("--device", choices=("pico", "meta"), required=True)
    parser.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="openarmv1")
    parser.add_argument(
        "--home-pose",
        default=None,
        help="Override a legacy named home pose. Omit to use the robot home_q.",
    )
    parser.add_argument("--side", choices=SIDE_CHOICES, default="both")
    add_teleop_motion_arguments(parser)
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=DEFAULT_TRANSLATION_SCALE,
        help="Scale HandUMI translation deltas before applying them to the robot TCP.",
    )
    parser.add_argument(
        "--tracking-stale-ms",
        type=float,
        default=DEFAULT_TRACKING_STALE_MS,
        help="Cancel and require re-anchoring if tracking stops advancing.",
    )
    parser.add_argument(
        "--gripper-park-hold-s",
        type=float,
        default=GRIPPER_PARK_HOLD_S,
        help="Seconds fully closed before the corresponding arm returns home.",
    )
    parser.add_argument(
        "--park-max-joint-speed-deg-s",
        type=float,
        default=DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S,
        help="Maximum joint speed while an arm returns home for standby.",
    )
    parser.add_argument(
        "--space-start",
        action="store_true",
        help="Allow keyboard Space to start any unanchored enabled arms.",
    )
    parser.add_argument(
        "--no-sounds", action="store_true", help="Disable spoken feedback."
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help="Override the robot/device Controller->TCP setup calibration.",
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local Feetech, tracking, and robot CAN configuration.",
    )

    parser.add_argument(
        "--cameras",
        type=_camera_list,
        default="left_wrist,right_wrist,workspace",
        help=(
            "Comma-separated cameras to connect and capture; defaults to "
            "left_wrist,right_wrist,workspace."
        ),
    )
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument(
        "--skip-cameras", action="store_true", help="Do not connect camera devices."
    )
    parser.add_argument(
        "--no-rerun",
        "--no-preview",
        dest="no_rerun",
        action="store_true",
        help="Disable the live camera preview.",
    )

    parser.add_argument("--feetech-port", type=str, default=None)
    parser.add_argument("--skip-feetech", action="store_true")

    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    pico_transport = parser.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Validate but do not auto-repair CAN with sudo before connecting.",
    )


def validate_physical_teleop_args(args: argparse.Namespace) -> None:
    """Validate safety constraints shared by physical teleop frontends."""
    if args.translation_scale <= 0.0:
        raise SystemExit("--translation-scale must be > 0.")
    if args.tracking_stale_ms <= 0.0:
        raise SystemExit("--tracking-stale-ms must be > 0.")
    if args.gripper_park_hold_s < 0.0:
        raise SystemExit("--gripper-park-hold-s must be >= 0.")
    if args.park_max_joint_speed_deg_s <= 0.0:
        raise SystemExit("--park-max-joint-speed-deg-s must be > 0.")
    if args.skip_feetech and not args.space_start:
        raise SystemExit(
            "--skip-feetech disables clap control; add --space-start so teleop can begin."
        )
    if args.cam_width <= 0 or args.cam_height <= 0 or args.cam_fps <= 0:
        raise SystemExit("--cam-width, --cam-height, and --cam-fps must be > 0.")


__all__ = [
    "DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S",
    "DEFAULT_TRACKING_STALE_MS",
    "DEFAULT_TRANSLATION_SCALE",
    "add_physical_teleop_arguments",
    "validate_physical_teleop_args",
]
