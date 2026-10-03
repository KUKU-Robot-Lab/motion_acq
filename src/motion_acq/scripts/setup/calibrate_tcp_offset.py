"""Calibrate controller -> physical HandUMI gripper TCP transforms from recorded poses.

The important transform is always:

    T_world_tcp = T_world_controller @ T_controller_tcp

That transform belongs to the physical tool assembly -- the controller mount
plus the gripper tip screwed onto HandUMI -- and not to any robot arm, so the
calibration files are named after that assembly:

    configs/calibration/controller_tcp/{device}_{tool}.yaml

Robots declare which assembly they were demonstrated with under
``handumi_tool`` in ``configs/robots/<robot>.yaml`` and point at the matching
file. Two robots sharing one physical tip share one file; one robot used with
two different tips needs two.

Calibration uses recorded pose7 controller data. Point ``--dataset`` at a
recording directory, or pass ``--parquet``/``--csv`` explicitly. Fits are
written to a candidate file by default, never straight into the project
calibration -- inspect and promote them deliberately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from motion_acq.calibration.control_tcp import (
    DEFAULT_PARQUET,
    DEFAULT_CALIBRATION_DIR,
    SIDES,
    SUPPORTED_DEVICES,
    existing_or_identity,
    load_controller_tcp_calibration,
    load_csv_poses,
    load_episode_poses,
    solve_orientation_offset,
    solve_pivot_offset,
    write_controller_tcp_calibration,
)
from motion_acq.robots.utils import IDENTITY_POSE7, pose_inv, quat_normalize

COMMANDS = {"pivot", "orient", "inspect", "promote"}


DEFAULT_CANDIDATE = Path("outputs/calibration/controller_tcp_candidate.yaml")

# Fall back to the same device `handumi record` assumes when the rig is silent.
# control_tcp.DEFAULT_DEVICE is PICO, and using it here made the two commands
# disagree about the same rig.
FALLBACK_DEVICE = "meta"
DEFAULT_CAPTURE_TIME_S = 25.0
DEFAULT_CAPTURE_RATE_HZ = 30.0
DEFAULT_TRACKING_LOSS_TIMEOUT_S = 3.0
PIVOT_RMS_LIMIT_M = 0.005
PIVOT_MAX_LIMIT_M = 0.010
PIVOT_CONDITION_LIMIT = 500.0
SYMMETRY_LIMIT_M = 0.005
DEVICE_FLIP_AXIS = {"pico": 0, "meta": 1}
AXIS_NAMES = ("x", "y", "z")


def _rig_device(rig_config: Path | None = None) -> str | None:
    """Tracking device from the operator's rig, so --device can be omitted."""
    from motion_acq.config import DEFAULT_RIG_CONFIG, load_optional_rig_section

    try:
        recording = load_optional_rig_section(
            rig_config or DEFAULT_RIG_CONFIG, "recording"
        )
    except SystemExit:
        return None
    device = recording.get("device")
    return str(device) if device in SUPPORTED_DEVICES else None


def _device(args: argparse.Namespace) -> str:
    return (
        args.device_local
        or args.device
        or _rig_device(getattr(args, "rig_config", None))
        or FALLBACK_DEVICE
    )


def _output_path(args: argparse.Namespace) -> Path:
    """Where a fit is written.

    Defaults to a candidate file rather than the project calibration: a fit
    has to clear the acceptance limits and be symmetrized before it is
    promoted, and pivot alone never determines orientation.
    """
    if args.output is not None:
        return args.output
    return DEFAULT_CANDIDATE


def dataset_parquet(dataset: Path) -> Path:
    """Locate the single parquet of a recording made by ``handumi record``."""
    matches = sorted(dataset.glob("data/chunk-*/file-*.parquet"))
    if not matches:
        raise SystemExit(
            f"No recording found under {dataset}. Expected "
            f"{dataset}/data/chunk-000/file-000.parquet from `handumi record`."
        )
    return matches[0]


def _load_input_poses(args: argparse.Namespace, side: str) -> np.ndarray:
    if args.csv is not None:
        return load_csv_poses(args.csv, side)
    if getattr(args, "dataset", None) is not None:
        parquet = dataset_parquet(args.dataset)
        episode = 0 if args.episode is None else args.episode
        return load_episode_poses(parquet, episode, side, column=args.column)
    if args.episode is None:
        raise SystemExit("Use --dataset, or --episode with --parquet, or --csv")
    return load_episode_poses(args.parquet, args.episode, side, column=args.column)


def _capture_output_dir(args: argparse.Namespace) -> Path:
    if args.capture_output_dir is not None:
        return args.capture_output_dir
    return Path("outputs") / f"tcp_pivot_{args.side}"


def _identity_tcp_calibration():
    from motion_acq.calibration.control_tcp import ControllerTcpCalibration

    pose = IDENTITY_POSE7.astype(np.float32)
    return ControllerTcpCalibration(left=pose.copy(), right=pose.copy(), source=None)


def _live_tracker(args: argparse.Namespace, device: str):
    """Open only VR tracking; TCP capture never initializes cameras or motors."""
    from motion_acq.config import DEFAULT_RIG_CONFIG
    from motion_acq.scripts.record import build_tracker

    tracker_args = SimpleNamespace(
        device=device,
        rig_config=args.rig_config or DEFAULT_RIG_CONFIG,
        quest_ip=args.quest_ip,
        tcp_port=args.tcp_port,
        sync_port=args.sync_port,
        pico_mode=args.pico_mode,
        pico_wifi=args.pico_wifi,
        skip_adb_check=args.skip_adb_check,
    )
    return build_tracker(
        tracker_args, _identity_tcp_calibration(), reset_workspace_on_x=False
    )


def _capture_rows(
    tracker: Any,
    *,
    side: str,
    duration_s: float,
    rate_hz: float = DEFAULT_CAPTURE_RATE_HZ,
    tracking_loss_timeout_s: float = DEFAULT_TRACKING_LOSS_TIMEOUT_S,
) -> list[dict[str, float | int | str]]:
    """Collect the selected raw device-controller pose and nothing else."""
    rows: list[dict[str, float | int | str]] = []
    started = time.monotonic()
    loss_started: float | None = None
    next_sample = started
    while True:
        now = time.monotonic()
        if now - started >= duration_s:
            break
        sample = tracker.latest()
        tracked = bool(sample.streaming and getattr(sample, f"{side}_device_tracked"))
        if tracked:
            loss_started = None
            pose = np.asarray(
                getattr(sample, f"{side}_device_controller_pose"), dtype=np.float32
            ).reshape(7)
            rows.append(
                {
                    "elapsed_s": now - started,
                    "side": side,
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2]),
                    "qx": float(pose[3]),
                    "qy": float(pose[4]),
                    "qz": float(pose[5]),
                    "qw": float(pose[6]),
                    "device_time_ns": int(sample.device_time_ns),
                    "pc_monotonic_ns": int(sample.pc_monotonic_ns),
                    "sequence": int(sample.sequence),
                }
            )
        else:
            if loss_started is None:
                loss_started = now
            if now - loss_started >= tracking_loss_timeout_s:
                raise SystemExit(
                    f"{side.capitalize()} controller tracking was lost for "
                    f"{tracking_loss_timeout_s:g} s; capture discarded."
                )
        next_sample += 1.0 / rate_hz
        time.sleep(max(0.0, next_sample - time.monotonic()))
    return rows


def _capture_live_poses(args: argparse.Namespace, device: str) -> np.ndarray:
    if args.time_s <= 0:
        raise SystemExit("--time-s must be greater than zero.")
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be greater than zero.")
    output_dir = _capture_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = _live_tracker(args, device)
    tracker.start()
    try:
        print(f"[tcp] Waiting for the {args.side} {device} controller ...")
        while True:
            sample = tracker.latest()
            if sample.streaming and getattr(sample, f"{args.side}_device_tracked"):
                break
            time.sleep(0.1)
        input(
            f"[tcp] Fix the {args.side} tip at the pivot, then press ENTER. "
            f"Capture will run for {args.time_s:g} s: "
        )
        print("[tcp] Recording VR pose only; rotate through varied orientations.")
        rows = _capture_rows(
            tracker, side=args.side, duration_s=args.time_s, rate_hz=args.rate_hz
        )
    except KeyboardInterrupt as exc:
        raise SystemExit("TCP capture cancelled; no fit was written.") from exc
    finally:
        tracker.stop()

    if len(rows) < 8:
        raise SystemExit("TCP capture produced fewer than 8 tracked frames.")
    csv_path = output_dir / "poses.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metadata = {
        "tracking_device": device,
        "side": args.side,
        "duration_s": args.time_s,
        "rate_hz": args.rate_hz,
        "samples": len(rows),
        "poses": str(csv_path),
    }
    (output_dir / "capture.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[tcp] saved VR-only capture: {csv_path} ({len(rows)} samples)")
    return load_csv_poses(csv_path, args.side)


def _existing_or_seeded(args: argparse.Namespace, output: Path) -> tuple[np.ndarray, np.ndarray]:
    return existing_or_identity(output)


def _save_side_pose(args: argparse.Namespace, side_pose: np.ndarray, *, update_rotation: bool) -> Path:
    output = _output_path(args)
    preserved_fits: dict[str, Any] = {}
    if output.exists():
        current_data = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        root = current_data.get("calibration", current_data)
        if isinstance(root, dict) and isinstance(root.get("pivot_fits"), dict):
            preserved_fits = root["pivot_fits"]
    left, right = _existing_or_seeded(args, output)
    target = left if args.side == "left" else right
    if update_rotation:
        target[3:] = quat_normalize(side_pose[3:])
    else:
        target[:3] = side_pose[:3]
    write_controller_tcp_calibration(output, left=left, right=right)
    if preserved_fits:
        data = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        data.setdefault("calibration", {})["pivot_fits"] = preserved_fits
        output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def _pivot_passes(result: Any) -> bool:
    return (
        result.rms_error < PIVOT_RMS_LIMIT_M
        and result.max_error < PIVOT_MAX_LIMIT_M
        and result.condition < PIVOT_CONDITION_LIMIT
    )


def _save_pivot_fit(output: Path, *, device: str, side: str, result: Any) -> None:
    data = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
    root = data.setdefault("calibration", {})
    fits = root.setdefault("pivot_fits", {})
    fits[side] = {
        "tracking_device": device,
        "samples": int(result.num_samples),
        "rms_error_m": float(result.rms_error),
        "max_error_m": float(result.max_error),
        "condition": float(result.condition),
        "accepted": _pivot_passes(result),
    }
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _print_pivot_report(device: str, side: str, result, output: Path) -> None:
    print(f"[{device}-tcp] side={side} samples={result.num_samples}")
    print(
        f"[{device}-tcp] controller->TCP position (m):",
        np.array2string(result.position, precision=5, suppress_small=True),
    )
    print(
        f"[{device}-tcp] fixed TCP point in tracking world (m):",
        np.array2string(result.pivot_world, precision=5, suppress_small=True),
    )
    print(
        f"[{device}-tcp] residual rms={result.rms_error * 100:.2f}cm "
        f"max={result.max_error * 100:.2f}cm condition={result.condition:.1f}"
    )
    if result.rms_error >= PIVOT_RMS_LIMIT_M or result.max_error >= PIVOT_MAX_LIMIT_M:
        print(f"[{device}-tcp] FAIL: high residual; the tip probably slipped.")
    if result.condition >= PIVOT_CONDITION_LIMIT:
        print(f"[{device}-tcp] WARNING: weak rotation diversity; rotate through more poses.")
    print(f"[{device}-tcp] fit: {'PASS' if _pivot_passes(result) else 'FAIL'}")
    print(f"[{device}-tcp] wrote: {output}")


def pivot_main(args: argparse.Namespace) -> None:
    device = _device(args)
    has_input = args.csv is not None or args.dataset is not None or args.episode is not None
    poses = _load_input_poses(args, args.side) if has_input else _capture_live_poses(args, device)
    result = solve_pivot_offset(poses)
    side_pose = IDENTITY_POSE7.copy()
    side_pose[:3] = result.position
    output = _save_side_pose(args, side_pose, update_rotation=False)
    _save_pivot_fit(output, device=device, side=args.side, result=result)
    _print_pivot_report(device, args.side, result, output)


def orient_main(args: argparse.Namespace) -> None:
    poses = _load_input_poses(args, args.side)
    quat = quat_normalize(np.asarray(args.tcp_quat_world, dtype=np.float32))
    offset_quat = solve_orientation_offset(poses, quat)
    side_pose = IDENTITY_POSE7.copy()
    side_pose[3:] = offset_quat
    output = _save_side_pose(args, side_pose, update_rotation=True)
    device = _device(args)
    print(f"[{device}-tcp] side={args.side} controller->TCP quaternion xyzw:")
    print("          ", np.array2string(offset_quat, precision=5, suppress_small=True))
    print(f"[{device}-tcp] wrote: {output}")


def _fit_passes_mapping(fit: dict[str, Any]) -> bool:
    try:
        return (
            float(fit["rms_error_m"]) < PIVOT_RMS_LIMIT_M
            and float(fit["max_error_m"]) < PIVOT_MAX_LIMIT_M
            and float(fit["condition"]) < PIVOT_CONDITION_LIMIT
        )
    except (KeyError, TypeError, ValueError):
        return False


def _fit_score(fit: dict[str, Any]) -> float:
    """Higher means closer to (or beyond) an individual acceptance limit."""
    return max(
        float(fit["rms_error_m"]) / PIVOT_RMS_LIMIT_M,
        float(fit["max_error_m"]) / PIVOT_MAX_LIMIT_M,
        float(fit["condition"]) / PIVOT_CONDITION_LIMIT,
    )


def _fit_device(fits: dict[str, Any]) -> str:
    devices = {
        str(fit.get("tracking_device"))
        for side in SIDES
        if isinstance((fit := fits.get(side)), dict) and fit.get("tracking_device")
    }
    if len(devices) != 1:
        raise SystemExit(
            "Candidate sides do not identify one common tracking device; "
            "recapture both sides with the same device."
        )
    device = devices.pop()
    if device not in DEVICE_FLIP_AXIS:
        raise SystemExit(f"Unsupported tracking device in candidate: {device!r}")
    return device


def _symmetry_errors(
    calibration, *, device: str
) -> np.ndarray:
    left = np.asarray(calibration.left[:3], dtype=np.float64)
    right = np.asarray(calibration.right[:3], dtype=np.float64)
    errors = np.abs(left - right)
    flip_axis = DEVICE_FLIP_AXIS[device]
    errors[flip_axis] = abs(left[flip_axis] + right[flip_axis])
    return errors


def _symmetrized_positions(calibration, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(calibration.left[:3], dtype=np.float64)
    right = np.asarray(calibration.right[:3], dtype=np.float64)
    result_left = (left + right) / 2.0
    result_right = result_left.copy()
    flip_axis = DEVICE_FLIP_AXIS[device]
    result_left[flip_axis] = (left[flip_axis] - right[flip_axis]) / 2.0
    result_right[flip_axis] = -result_left[flip_axis]
    return result_left, result_right


def _print_symmetry_report(calibration, fits: dict[str, Any]) -> bool:
    device = _fit_device(fits)
    errors = _symmetry_errors(calibration, device=device)
    passed = bool(np.all(errors < SYMMETRY_LIMIT_M))
    flip_name = AXIS_NAMES[DEVICE_FLIP_AXIS[device]]
    print(
        f"  bilateral symmetry ({device}, sign-flip axis={flip_name}, "
        f"limit <{SYMMETRY_LIMIT_M * 1000:.1f}mm):"
    )
    for axis, error in zip(AXIS_NAMES, errors, strict=True):
        relation = "opposite signs" if axis == flip_name else "same value"
        status = "PASS" if error < SYMMETRY_LIMIT_M else "FAIL"
        print(f"    {axis}: {status} mismatch={error * 1000:.2f}mm ({relation})")
    print(f"    overall: {'PASS' if passed else 'FAIL'}")
    if not passed:
        left_fit = fits["left"]
        right_fit = fits["right"]
        recommended = (
            "left" if _fit_score(left_fit) >= _fit_score(right_fit) else "right"
        )
        print(
            f"    Recommended first recapture: {recommended} "
            "(weaker individual fit metrics)."
        )
        print(
            "    Symmetry alone cannot prove which side drifted; recapture the "
            "other side too if the mismatch remains."
        )
    return passed


def inspect_main(args: argparse.Namespace) -> None:
    path = args.path or _output_path(args)
    calibration = load_controller_tcp_calibration(path)
    print(f"[tcp] loaded: {path}")
    for side, pose in (("left", calibration.left), ("right", calibration.right)):
        inv_pose = pose_inv(pose)
        print(f"  {side}:")
        print("    controller->tcp:", np.array2string(pose, precision=5, suppress_small=True))
        print("    tcp->controller:", np.array2string(inv_pose, precision=5, suppress_small=True))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = data.get("calibration", data)
    fits = root.get("pivot_fits", {}) if isinstance(root, dict) else {}
    if fits:
        print("  pivot fits (limits: RMS <0.50cm, max <1.00cm, condition <500):")
        for side in SIDES:
            fit = fits.get(side)
            if not isinstance(fit, dict):
                print(f"    {side}: NOT CAPTURED")
                continue
            print(
                f"    {side}: {'PASS' if _fit_passes_mapping(fit) else 'FAIL'} "
                f"rms={float(fit['rms_error_m']) * 100:.2f}cm "
                f"max={float(fit['max_error_m']) * 100:.2f}cm "
                f"condition={float(fit['condition']):.1f}"
            )
        if all(isinstance(fits.get(side), dict) for side in SIDES):
            _print_symmetry_report(calibration, fits)


def _calibration_filename(value: str) -> Path:
    path = Path(value)
    if path.name != value or path.suffix != ".yaml":
        raise argparse.ArgumentTypeError(
            "use a .yaml filename only, without a directory"
        )
    return path


def _replace_positions_in_yaml(
    text: str, *, left: np.ndarray, right: np.ndarray
) -> str:
    """Replace only position scalars, preserving comments and all other YAML."""
    lines = text.splitlines(keepends=True)
    side_positions = {"left": left, "right": right}
    current_side: str | None = None
    replaced: set[str] = set()
    side_pattern = re.compile(r"^(\s+)(left|right):\s*(?:#.*)?$")
    position_pattern = re.compile(r"^(\s+)position:\s*(?:#.*)?$")
    index = 0
    while index < len(lines):
        stripped = lines[index].rstrip("\r\n")
        side_match = side_pattern.match(stripped)
        if side_match:
            current_side = side_match.group(2)
            index += 1
            continue
        position_match = position_pattern.match(stripped)
        if position_match and current_side in side_positions:
            indent = position_match.group(1) + "  "
            newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
            values = side_positions[current_side]
            replacement = [f"{indent}- {float(value):.9g}{newline}" for value in values]
            if index + 3 >= len(lines) or not all(
                lines[row].lstrip().startswith("-")
                for row in range(index + 1, index + 4)
            ):
                raise SystemExit(
                    f"Cannot safely update {current_side}.position: expected three YAML list items."
                )
            lines[index + 1 : index + 4] = replacement
            replaced.add(current_side)
            if replaced == set(SIDES):
                return "".join(lines)
            index += 4
            continue
        index += 1
    if replaced != set(SIDES):
        raise SystemExit(
            "Target does not contain standard left/right controller TCP positions."
        )
    return "".join(lines)


def _new_target_yaml(
    source: Path, *, left: np.ndarray, right: np.ndarray
) -> str:
    source_calibration = load_controller_tcp_calibration(source)
    left_pose = source_calibration.left.copy()
    right_pose = source_calibration.right.copy()
    left_pose[:3] = left
    right_pose[:3] = right
    data = {
        "calibration": {
            "frame_convention": "pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion",
            "controller_to_gripper_tcp": {
                "left": {
                    "position": [float(value) for value in left_pose[:3]],
                    "quaternion": [float(value) for value in left_pose[3:]],
                },
                "right": {
                    "position": [float(value) for value in right_pose[:3]],
                    "quaternion": [float(value) for value in right_pose[3:]],
                },
            },
        }
    }
    return yaml.safe_dump(data, sort_keys=False)


def promote_main(args: argparse.Namespace) -> None:
    candidate_path = args.candidate or DEFAULT_CANDIDATE
    candidate = load_controller_tcp_calibration(candidate_path)
    candidate_data = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
    root = candidate_data.get("calibration", candidate_data)
    fits = root.get("pivot_fits", {}) if isinstance(root, dict) else {}
    if not all(isinstance(fits.get(side), dict) for side in SIDES):
        raise SystemExit("Candidate needs pivot fit metrics for both sides.")
    failed = [side for side in SIDES if not _fit_passes_mapping(fits[side])]
    if failed:
        raise SystemExit(
            f"Cannot promote: individual pivot fit failed for {', '.join(failed)}."
        )
    device = _fit_device(fits)
    errors = _symmetry_errors(candidate, device=device)
    if np.any(errors >= SYMMETRY_LIMIT_M):
        _print_symmetry_report(candidate, fits)
        if not args.force:
            raise SystemExit("Cannot promote: bilateral symmetry check failed.")
        print(
            "WARNING: forcing promotion despite failed bilateral symmetry; "
            "positions will be symmetrized."
        )

    target = DEFAULT_CALIBRATION_DIR / args.target
    if not (args.target.name == f"{device}.yaml" or args.target.name.startswith(f"{device}_")):
        raise SystemExit(
            f"Target {args.target.name!r} does not match candidate device {device!r}."
        )
    source = (
        DEFAULT_CALIBRATION_DIR / args.quaternion_source
        if args.quaternion_source is not None
        else None
    )
    left, right = _symmetrized_positions(candidate, device=device)
    if target.exists():
        if not args.yes:
            answer = input(f"{target} already exists. Override it? [Y/n] ").strip().lower()
            if answer not in {"", "y", "yes"}:
                print("Promotion cancelled.")
                return
        updated = _replace_positions_in_yaml(
            target.read_text(encoding="utf-8"), left=left, right=right
        )
    else:
        if source is None:
            raise SystemExit(
                "A new target has no official quaternions. Pass "
                "--quaternion-source EXISTING.yaml to copy orientation from the "
                "same tracking device and controller mount."
            )
        if not source.exists():
            raise SystemExit(f"Quaternion source not found: {source}")
        if not (
            args.quaternion_source.name == f"{device}.yaml"
            or args.quaternion_source.name.startswith(f"{device}_")
        ):
            raise SystemExit(
                "Quaternion source must use the same tracking device as the candidate."
            )
        updated = _new_target_yaml(source, left=left, right=right)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")
    print(f"Promoted {device} controller TCP positions to: {target}")
    if source is None:
        print("Quaternions and all other target values were preserved.")
    else:
        print(f"Quaternions were copied from: {source}")
    print("  left.position: ", np.array2string(left, precision=6))
    print("  right.position:", np.array2string(right, precision=6))


def solve_pivot(
    controller_positions: np.ndarray,
    controller_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Legacy test/helper wrapper around :func:`solve_pivot_offset`."""
    positions = np.asarray(controller_positions, dtype=np.float32)
    rotations = np.asarray(controller_rotations, dtype=np.float64)
    quats = Rotation.from_matrix(rotations).as_quat().astype(np.float32)
    poses = np.concatenate([positions, quats], axis=1)
    result = solve_pivot_offset(poses)
    return result.position, result.pivot_world, result.rms_error


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default=None, dest="device_local")


def add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Recording directory from `handumi record`; its parquet and "
        "first episode are resolved automatically.",
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--csv", type=Path, help="CSV with x,y,z,qx,qy,qz,qw and optional side")
    parser.add_argument("--column", help="Override parquet pose column")
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--output", type=Path, default=None)
    add_device_arg(parser)


def add_live_capture_args(parser: argparse.ArgumentParser) -> None:
    from motion_acq.config import DEFAULT_RIG_CONFIG

    parser.add_argument(
        "--time-s", type=float, default=DEFAULT_CAPTURE_TIME_S,
        help="Live VR-only pivot capture duration.",
    )
    parser.add_argument(
        "--output-dir", dest="capture_output_dir", type=Path,
        help="Capture directory (default: outputs/tcp_pivot_<side>).",
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_CAPTURE_RATE_HZ)
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--pico-adb", action="store_true")
    transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")


def _argv_with_default_command(argv: list[str]) -> list[str]:
    if not argv or "-h" in argv or "--help" in argv:
        return argv
    if any(arg in COMMANDS for arg in argv):
        return argv
    return ["pivot", *argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pivot = sub.add_parser(
        "pivot",
        help="Estimate controller->TCP translation by keeping the gripper tip fixed.",
    )
    add_common_input_args(pivot)
    add_live_capture_args(pivot)
    pivot.set_defaults(func=pivot_main)

    orient = sub.add_parser(
        "orient",
        help="Estimate controller->TCP rotation from a known TCP world orientation.",
    )
    add_common_input_args(orient)
    orient.add_argument(
        "--tcp-quat-world",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        required=True,
        help="Desired TCP orientation in the same world frame as the recorded controller poses.",
    )
    orient.set_defaults(func=orient_main)

    inspect = sub.add_parser("inspect", help="Print a calibration YAML.")
    inspect.add_argument("path", type=Path, nargs="?")
    inspect.add_argument("--output", type=Path, default=None)
    add_device_arg(inspect)
    inspect.set_defaults(func=inspect_main)

    promote = sub.add_parser(
        "promote",
        help="Validate, symmetrize, and promote candidate positions.",
    )
    promote.add_argument(
        "--target",
        type=_calibration_filename,
        required=True,
        help=f"YAML filename under {DEFAULT_CALIBRATION_DIR}.",
    )
    promote.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help=f"Candidate YAML (default: {DEFAULT_CANDIDATE}).",
    )
    promote.add_argument(
        "--quaternion-source",
        type=_calibration_filename,
        help="Existing YAML whose quaternions seed a new target.",
    )
    promote.add_argument(
        "--yes",
        action="store_true",
        help="Override an existing target without prompting.",
    )
    promote.add_argument(
        "--force",
        action="store_true",
        help="Promote and symmetrize even when bilateral symmetry fails.",
    )
    promote.set_defaults(func=promote_main)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_argv_with_default_command(sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
