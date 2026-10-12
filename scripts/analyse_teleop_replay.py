#!/usr/bin/env python3
"""Replay recorded operator hand poses through the live teleop arm pipeline.

10.12 user: the arms must follow the operator's hand as far as they can, and
the arm control decides how. This replays the operator's hand path (a teleop
recording or a ``teleop-real --joint-debug`` trace) through the same
TeleopController + incremental DLS the robot runs, once per variant, and
compares each with the operator's own (uncapped) hand target.

Run on the station that holds the data, e.g. on arm4090:
  .venv/bin/python scripts/analyse_teleop_replay.py \
      --dataset outputs/arm4090_20261012_103629 --scale 1.0
  .venv/bin/python scripts/analyse_teleop_replay.py \
      --trace logs/teleop_trace/trace_left_20261011_220739.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from motion_acq.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
)
from motion_acq.robots.kinematics import rotation_error_deg
from motion_acq.robots.registry import _rotation_limits, load_embodiment
from motion_acq.robots.utils import pose_mul
from motion_acq.teleop.common import tracking_world_map
from motion_acq.teleop.core import SIDES, TeleopController
from motion_acq.teleop.dls import DlsConfig, make_real_teleop_dls_solver

CONTROL_HZ = 72.0
ROBOT = "openarm_rh56f1"


@dataclass(frozen=True)
class Frame:
    t: float
    source: dict[str, np.ndarray]
    tracked: dict[str, bool]
    anchor: dict[str, float]  # side -> heading (rad) when the arm starts on this frame


@dataclass(frozen=True)
class Variant:
    name: str
    caps: bool
    dls: DlsConfig
    speed_cap_rad_s: float | None = None  # None: the live cap (1 rad/s)
    substeps: int = 1  # DLS steps per control tick (many = converged, no lag)


# The 10.11 hand-turn caps, removed from the robot YAML on 10.12.
CAPS_1011 = {
    "knee": 0.6,
    "left": {"pos_deg": [85, 80, 50], "neg_deg": [40, 30, 45]},
    "right": {"pos_deg": [40, 80, 45], "neg_deg": [85, 30, 50]},
}

VARIANTS = (
    Variant("10.11 (caps, plain DLS)", True, DlsConfig(limit_aware=False)),
    Variant("caps, limit-aware", True, DlsConfig()),
    Variant("no caps, plain DLS", False, DlsConfig(limit_aware=False)),
    Variant("no caps, limit-aware", False, DlsConfig()),
    Variant("no caps, limit-aware, 2 rad/s", False, DlsConfig(), speed_cap_rad_s=2.0),
    Variant("reference: converged, no speed cap", False, DlsConfig(),
            speed_cap_rad_s=100.0, substeps=30),
)


def _yaw_map(world_map: np.ndarray, heading: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    undo = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(world_map, dtype=np.float64) @ undo).astype(np.float32)


def load_dataset(root: Path, episode: int) -> tuple[list[Frame], np.ndarray]:
    import pandas as pd

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[df.episode_index == episode].sort_values("frame_index")
    if df.empty:
        raise SystemExit(f"episode {episode} not in {root}")
    workspace = np.stack(df["observation.tracking.workspace_from_device_pose"].values)
    heading = np.stack(df["observation.tracking.anchor_heading_rad"].values)
    frames: list[Frame] = []
    was = {"left": False, "right": False}
    for i in range(len(df)):
        source, tracked, anchor = {}, {}, {}
        for k, side in enumerate(SIDES):
            device = np.asarray(df.iloc[i][f"observation.tracking.{side}_device_controller_pose"])
            source[side] = pose_mul(workspace[i], device).astype(np.float32)
            tracked[side] = bool(df.iloc[i][f"observation.tracking.{side}_tracked"])
            now = bool(np.isfinite(heading[i, k]))
            if now and not was[side]:
                anchor[side] = float(heading[i, k])
            was[side] = now
        frames.append(Frame(float(df.iloc[i]["timestamp"]), source, tracked, anchor))
    action = np.stack(df["action"].values).astype(np.float32)
    info = json.loads((root / "meta" / "info.json").read_text())
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
    runtime_names = list(load_embodiment(ROBOT).joint_names)
    order = [names.index(n) for n in runtime_names if n in names]
    if len(order) != len(names):
        raise SystemExit("action joints do not match the robot's joints")
    return frames, action[:, order]  # in the robot's joint order


def load_trace(path: Path) -> tuple[list[Frame], None]:
    frames: list[Frame] = []
    pending: dict[str, float] = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["kind"] == "anchor":
            pending[rec["side"]] = math.radians(rec["heading_deg"])
        elif rec["kind"] == "frame":
            source = {s: np.asarray(v, dtype=np.float32) for s, v in rec["src"].items() if v}
            tracked = {s: bool(rec["tracked"].get(s, False)) for s in SIDES}
            for side in SIDES:
                source.setdefault(side, np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            frames.append(Frame(rec["ts"] * 1e-9, source, tracked, dict(pending)))
            pending = {}
    return frames, None


def resample(frames: list[Frame]) -> list[Frame]:
    """Hold each tracking frame for the control ticks until the next (72 Hz IK)."""
    out: list[Frame] = []
    t0 = frames[0].t
    tick = 0
    for i, frame in enumerate(frames):
        end = frames[i + 1].t if i + 1 < len(frames) else frame.t + 1.0 / CONTROL_HZ
        first = True
        while t0 + tick / CONTROL_HZ < end:
            out.append(frame if first else replace(frame, anchor={}))
            first = False
            tick += 1
    return out


def run(variant: Variant, frames: list[Frame], scale: float, sides: tuple[str, ...]) -> dict:
    runtime = load_embodiment(ROBOT)
    home_q = runtime.home_q()
    controller = TeleopController(
        runtime, home_q=home_q, enabled_sides=sides,
        source_world_to_robot_world=tracking_world_map("meta"),
        translation_scale=scale, heading_from_hmd=True,
    )
    controller.rotation_limits = _rotation_limits(CAPS_1011) if variant.caps else {}
    controller.solver = make_real_teleop_dls_solver(
        runtime, controller.solver, home_q, input_rate_hz=CONTROL_HZ, config=variant.dls,
    )
    controller.solver.set_timestep(1.0 / CONTROL_HZ)
    if variant.speed_cap_rad_s is not None:
        for side, caps in controller.solver.side_joint_speed_limits_rad_s.items():
            controller.solver.side_joint_speed_limits_rad_s[side] = np.full_like(
                caps, variant.speed_cap_rad_s)
    base_map = tracking_world_map("meta")
    q_log, raw, got, active = [], {s: [] for s in SIDES}, {s: [] for s in SIDES}, []
    for frame in frames:
        for side, heading in frame.anchor.items():
            if side not in sides:
                continue
            world_map = _yaw_map(base_map, heading)
            src = frame.source[side]
            controller.anchors[side] = {
                "source": src.copy(),
                "adapter": local_frame_adapter(src, controller.anchor_ref[side], world_map),
                "heading": np.float32(heading),
            }
        for _ in range(variant.substeps - 1):
            controller.step(frame.source, frame.tracked, {"left": 0.0, "right": 0.0})
        step = controller.step(frame.source, frame.tracked, {"left": 0.0, "right": 0.0})
        q_log.append(step.q.copy())
        fk = dict(zip(SIDES, controller.solver.fk_pose7(step.q), strict=True))
        on = []
        for side in SIDES:
            anchor = controller.anchors[side]
            if anchor is None or not frame.tracked[side]:
                raw[side].append(np.full(7, np.nan, dtype=np.float32))
                got[side].append(fk[side])
                continue
            raw[side].append(local_relative_robot_target_pose7(
                previous_source_pose7=anchor["source"], current_source_pose7=frame.source[side],
                base_robot_pose7=controller.anchor_ref[side], adapter_rot=anchor["adapter"],
                home_robot_pose7=controller.anchor_ref[side], translation_scale=scale,
                max_reach=controller.max_reach,
            ))
            got[side].append(fk[side])
            on.append(side)
        active.append(on)
    return {
        "q": np.asarray(q_log), "raw": {s: np.asarray(v) for s, v in raw.items()},
        "got": {s: np.asarray(v) for s, v in got.items()}, "active": active,
        "side_indices": controller.side_indices, "home_q": home_q,
    }


def summarize(name: str, result: dict, sides: tuple[str, ...]) -> list[str]:
    rows = []
    q = result["q"]
    for side in sides:
        mask = np.array([side in a for a in result["active"]])
        if not mask.any():
            continue
        raw, got = result["raw"][side][mask], result["got"][side][mask]
        pos = np.linalg.norm(raw[:, :3] - got[:, :3], axis=1) * 100.0
        rot = rotation_error_deg(raw, got)
        idx = result["side_indices"][side]
        qs = np.degrees(q[mask][:, idx])
        shoulder = qs[:, :2]
        speed = np.abs(np.diff(qs, axis=0)).max(axis=1) * CONTROL_HZ
        rows.append(
            f"| {name} | {side} | {np.percentile(pos, 95):.1f} / {pos.max():.1f} | "
            f"{np.percentile(rot, 95):.0f} / {rot.max():.0f} | "
            f"{np.ptp(shoulder[:, 0]):.0f} / {np.ptp(shoulder[:, 1]):.0f} | "
            f"{np.percentile(speed, 99):.0f} |"
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path)
    src.add_argument("--trace", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None, help="translation scale of the run")
    ap.add_argument("--save", type=Path, default=None, help="npz with every variant's joints")
    args = ap.parse_args()
    if args.dataset:
        frames, action = load_dataset(args.dataset, args.episode)
        sides = SIDES
        scale = 0.5 if args.scale is None else args.scale
    else:
        frames, action = load_trace(args.trace)
        setup = json.loads(args.trace.read_text().splitlines()[0])
        sides = (setup["side"],) if setup["side"] in SIDES else SIDES
        scale = float(setup["translation_scale"]) if args.scale is None else args.scale
    ticks = resample(frames)
    print(f"{len(frames)} frames -> {len(ticks)} control ticks, scale {scale}, sides {sides}")
    print("| variant | arm | pos err p95 / max (cm) | turn err p95 / max (deg) | "
          "shoulder j1 / j2 range (deg) | fastest joint p99 (deg/s) |")
    print("|---|---|---|---|---|---|")
    saved = {}
    for variant in VARIANTS:
        result = run(variant, ticks, scale, sides)
        for row in summarize(variant.name, result, sides):
            print(row)
        saved[variant.name] = result["q"]
        if action is not None and variant is VARIANTS[0]:
            per_frame = np.searchsorted(
                np.array([t.t for t in ticks]), np.array([f.t for f in frames]), side="right") - 1
            replay = result["q"][np.clip(per_frame, 0, len(ticks) - 1)]
            idx = np.concatenate([result["side_indices"][s] for s in sides])
            diff = np.degrees(np.abs(replay[:, idx] - action[:, idx]))
            print(f"  check: 10.11 replay vs recorded action, median {np.median(diff):.1f} deg, "
                  f"p95 {np.percentile(diff, 95):.1f} deg")
    if args.save:
        np.savez(args.save, **{k.replace(" ", "_"): v for k, v in saved.items()})


if __name__ == "__main__":
    main()
