"""Shared motion-pipeline configuration for every live teleop frontend."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from motion_acq.teleop.common import DEFAULT_TELEOP_FPS, AdaptiveJointFilter
from motion_acq.teleop.trajectory import TeleopCommandStream

DEFAULT_COMMAND_RATE_HZ = 100.0
# One 30 Hz frame plus a small scheduling margin gives the 100 Hz player two
# real IK endpoints to interpolate. Prediction is restricted to 5 ms, so the
# normal path is interpolation rather than speculative controller motion.
DEFAULT_TRAJECTORY_DELAY_MS = 40.0
DEFAULT_MAX_EXTRAPOLATION_MS = 5.0
DEFAULT_COMMAND_EMA_TIME_CONSTANT_S = 0.0
# One Euro parameters at the 30 Hz IK cadence. The low stationary cutoff
# rejects solver/tracker chatter; velocity adaptation restores bandwidth as
# soon as an operator intentionally moves a joint.
DEFAULT_JOINT_FILTER_MIN_CUTOFF_HZ = 3.0
DEFAULT_JOINT_FILTER_VELOCITY_COEFFICIENT = 3.0
DEFAULT_JOINT_FILTER_DERIVATIVE_CUTOFF_HZ = 5.0


@dataclass(frozen=True)
class TeleopMotionConfig:
    """Rates and filters shared by sim, real, and recording teleoperation."""

    input_rate_hz: float = DEFAULT_TELEOP_FPS
    command_rate_hz: float = DEFAULT_COMMAND_RATE_HZ
    trajectory_delay_s: float = DEFAULT_TRAJECTORY_DELAY_MS / 1000.0
    max_extrapolation_s: float = DEFAULT_MAX_EXTRAPOLATION_MS / 1000.0
    command_ema_time_constant_s: float = DEFAULT_COMMAND_EMA_TIME_CONSTANT_S
    joint_filter_min_cutoff_hz: float = DEFAULT_JOINT_FILTER_MIN_CUTOFF_HZ
    joint_filter_velocity_coefficient: float = DEFAULT_JOINT_FILTER_VELOCITY_COEFFICIENT
    joint_filter_derivative_cutoff_hz: float = DEFAULT_JOINT_FILTER_DERIVATIVE_CUTOFF_HZ

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        input_rate_hz: float | None = None,
    ) -> TeleopMotionConfig:
        """Build normalized SI-unit configuration from shared CLI arguments."""
        return cls(
            input_rate_hz=float(
                args.fps if input_rate_hz is None else input_rate_hz
            ),
            command_rate_hz=float(args.command_rate_hz),
            trajectory_delay_s=float(args.trajectory_delay_ms) / 1000.0,
            max_extrapolation_s=float(args.max_extrapolation_ms) / 1000.0,
            command_ema_time_constant_s=float(args.motion_smoothing_time_constant_s),
            joint_filter_min_cutoff_hz=float(args.joint_filter_min_cutoff_hz),
            joint_filter_velocity_coefficient=float(
                args.joint_filter_velocity_coefficient
            ),
            joint_filter_derivative_cutoff_hz=float(
                args.joint_filter_derivative_cutoff_hz
            ),
        )

    def make_joint_filter(
        self,
        *,
        filtered_indices: tuple[int, ...] | None = None,
    ) -> AdaptiveJointFilter:
        """Create the responsive low-pass applied to fresh IK targets."""
        return AdaptiveJointFilter(
            sample_rate_hz=self.input_rate_hz,
            min_cutoff_hz=self.joint_filter_min_cutoff_hz,
            velocity_coefficient=self.joint_filter_velocity_coefficient,
            derivative_cutoff_hz=self.joint_filter_derivative_cutoff_hz,
            filtered_indices=filtered_indices,
        )

    def make_command_stream(
        self,
        write: Callable[[np.ndarray, dict[str, float]], None],
    ) -> TeleopCommandStream:
        """Create the common interpolated, fixed-rate output stage."""
        return TeleopCommandStream(
            write,
            command_rate_hz=self.command_rate_hz,
            delay_s=self.trajectory_delay_s,
            max_extrapolation_s=self.max_extrapolation_s,
            ema_time_constant_s=self.command_ema_time_constant_s,
        )


def add_teleop_motion_arguments(
    parser: argparse.ArgumentParser,
    *,
    help_transform: Callable[[str], str] | None = None,
) -> None:
    """Add the identical live-motion controls to a teleop CLI parser."""
    transform = help_transform or (lambda text: text)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_TELEOP_FPS,
        help=transform("Tracking/IK processing frequency."),
    )
    parser.add_argument(
        "--command-rate-hz",
        type=float,
        default=DEFAULT_COMMAND_RATE_HZ,
        help=transform("Fixed-rate playback frequency for interpolated joints."),
    )
    parser.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=DEFAULT_TRAJECTORY_DELAY_MS,
        help=transform(
            "Small playback window used to interpolate adjacent IK results."
        ),
    )
    parser.add_argument(
        "--max-extrapolation-ms",
        type=float,
        default=DEFAULT_MAX_EXTRAPOLATION_MS,
        help=transform(
            "Maximum constant-velocity bridge for a late tracking/IK sample."
        ),
    )
    parser.add_argument(
        "--motion-smoothing-time-constant-s",
        type=float,
        default=DEFAULT_COMMAND_EMA_TIME_CONSTANT_S,
        help=transform("100 Hz joint-command EMA time constant; 0 disables it."),
    )
    parser.add_argument(
        "--joint-filter-min-cutoff-hz",
        type=float,
        default=DEFAULT_JOINT_FILTER_MIN_CUTOFF_HZ,
        help=transform("Stationary IK joint filter cutoff; lower rejects more jitter."),
    )
    parser.add_argument(
        "--joint-filter-velocity-coefficient",
        type=float,
        default=DEFAULT_JOINT_FILTER_VELOCITY_COEFFICIENT,
        help=transform(
            "Increase joint-filter bandwidth in proportion to motion speed."
        ),
    )
    parser.add_argument(
        "--joint-filter-derivative-cutoff-hz",
        type=float,
        default=DEFAULT_JOINT_FILTER_DERIVATIVE_CUTOFF_HZ,
        help=transform("Cutoff used to estimate intentional joint velocity."),
    )


def validate_teleop_motion_args(args: argparse.Namespace) -> None:
    """Validate the shared motion arguments with CLI-friendly errors."""
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.command_rate_hz <= 0.0:
        raise SystemExit("--command-rate-hz must be > 0.")
    if args.trajectory_delay_ms < 0.0:
        raise SystemExit("--trajectory-delay-ms must be >= 0.")
    if args.max_extrapolation_ms < 0.0:
        raise SystemExit("--max-extrapolation-ms must be >= 0.")
    if args.motion_smoothing_time_constant_s < 0.0:
        raise SystemExit("--motion-smoothing-time-constant-s must be >= 0.")
    if args.joint_filter_min_cutoff_hz <= 0.0:
        raise SystemExit("--joint-filter-min-cutoff-hz must be > 0.")
    if args.joint_filter_velocity_coefficient < 0.0:
        raise SystemExit("--joint-filter-velocity-coefficient must be >= 0.")
    if args.joint_filter_derivative_cutoff_hz <= 0.0:
        raise SystemExit("--joint-filter-derivative-cutoff-hz must be > 0.")


__all__ = [
    "DEFAULT_COMMAND_EMA_TIME_CONSTANT_S",
    "DEFAULT_COMMAND_RATE_HZ",
    "DEFAULT_JOINT_FILTER_DERIVATIVE_CUTOFF_HZ",
    "DEFAULT_JOINT_FILTER_MIN_CUTOFF_HZ",
    "DEFAULT_JOINT_FILTER_VELOCITY_COEFFICIENT",
    "DEFAULT_MAX_EXTRAPOLATION_MS",
    "DEFAULT_TRAJECTORY_DELAY_MS",
    "TeleopMotionConfig",
    "add_teleop_motion_arguments",
    "validate_teleop_motion_args",
]
