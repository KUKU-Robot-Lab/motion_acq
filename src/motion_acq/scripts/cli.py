#!/usr/bin/env python3
"""Command router for motion_acq (Meta Quest -> OpenArm teleop and capture)."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Command:
    module: str
    description: str


COMMANDS = {
    ("doctor",): Command("motion_acq.scripts.doctor", "Check recording readiness"),
    ("tracking", "pose"): Command(
        "motion_acq.scripts.setup.print_controller_pose", "Print live controller poses"
    ),
    ("calibrate", "tcp"): Command(
        "motion_acq.scripts.setup.calibrate_tcp_offset", "Calibrate controller-to-TCP"
    ),
    ("teleop",): Command(
        "motion_acq.scripts.teleop_sim",
        "Teleoperate in simulation",
    ),
    ("teleop-real",): Command(
        "motion_acq.scripts.teleop_real",
        "Teleoperate a physical robot",
    ),
    ("teleop-record",): Command(
        "motion_acq.scripts.teleop_record",
        "Record real-robot teleoperation demonstrations",
    ),
    ("head",): Command(
        "motion_acq.scripts.head_teleop",
        "Meta Quest HMD -> Dynamixel pan/tilt head (fake bus by default)",
    ),
}


def _program_name() -> str:
    name = Path(sys.argv[0]).name
    return name if name in {"macq", "motion-acq"} else "macq"


def _print_help(
    prefix: tuple[str, ...] = (), *, program: str = "macq"
) -> None:
    label_prefix = f"{program} {' '.join(prefix)}" if prefix else program
    print(f"usage: {label_prefix} <command> [options]\n")
    print("Commands:")
    commands = {
        path: command
        for path, command in COMMANDS.items()
        if path[: len(prefix)] == prefix and len(path) > len(prefix)
    }
    labels = [" ".join(path) for path in commands]
    width = max(len(label) for label in labels)
    for (path, command), label in zip(commands.items(), labels, strict=True):
        print(f"  {program} {label:<{width}}  {command.description}")
    print(f"\nRun '{program} <command> --help' for command-specific options.")


def main(argv: list[str] | None = None) -> None:
    program = _program_name()
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        _print_help(program=program)
        return
    group = (values[0],)
    group_exists = any(
        path[:1] == group and len(path) > 1 for path in COMMANDS
    ) and group not in COMMANDS
    if group_exists and (len(values) == 1 or values[1] in {"-h", "--help"}):
        _print_help(group, program=program)
        return
    match = next(
        (
            (path, command)
            for path, command in sorted(
                COMMANDS.items(), key=lambda item: len(item[0]), reverse=True
            )
            if tuple(values[: len(path)]) == path
        ),
        None,
    )
    if match is None:
        requested = " ".join(values[:2])
        raise SystemExit(
            f"Unknown motion_acq command {requested!r}. Run '{program} --help'."
        )
    path, command = match
    rest = values[len(path) :]
    if path in {("teleop-real",), ("teleop-record",)}:
        # This IK is a tiny latency-sensitive dense solve. On the supported
        # NVIDIA setup, CPU execution has far lower tail latency than CUDA.
        # Respect an explicit user override for benchmarking/debugging.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    module = importlib.import_module(command.module)
    previous_argv = sys.argv
    try:
        sys.argv = [f"{program} {' '.join(path)}", *rest]
        module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
