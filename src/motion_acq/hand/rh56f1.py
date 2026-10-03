"""RH56F1 joint radians <-> vendor angle registers (driver slot order).

Port of sim2real policy_control/rh56f1_map.py (3ca354e), angle part only:
same end points, same per-side calibration and clipping. Python 3.10 safe so
the ROS hand node can import it on the system interpreter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

N_SLOTS = 6
LEAVE = -1  # SetAngle1: leave this axis where it is
DEFAULT_MAP = Path(__file__).resolve().parents[3] / "configs" / "hands" / "rh56f1_hand_map.yaml"


class HandMapError(ValueError):
    pass


@dataclass(frozen=True)
class Rh56f1Axis:
    name: str
    slot: int
    rad: tuple[float, float]
    reg: tuple[float, float]
    verified: bool
    cmd: tuple[int, int] | None = None  # command clip range (calibrated axes)

    def to_reg(self, q: float) -> int:
        lo, hi = self.rad
        q = min(max(float(q), min(lo, hi)), max(lo, hi))
        t = (q - lo) / (hi - lo)
        value = int(round(self.reg[0] + t * (self.reg[1] - self.reg[0])))
        if self.cmd is not None:
            value = min(max(value, self.cmd[0]), self.cmd[1])
        return value

    def to_rad(self, register: float) -> float:
        a, b = self.reg
        r = min(max(float(register), min(a, b)), max(a, b))
        return self.rad[0] + (r - a) / (b - a) * (self.rad[1] - self.rad[0])


@dataclass(frozen=True)
class Rh56f1Map:
    joint_order: tuple[str, ...]
    axes: tuple[Rh56f1Axis, ...]  # joint_order
    side_axes: Mapping[str, tuple[Rh56f1Axis, ...]]

    def axes_of(self, side: str | None) -> tuple[Rh56f1Axis, ...]:
        return self.side_axes.get(side, self.axes) if side else self.axes

    def to_registers(self, q: Mapping[str, float], *, side: str | None) -> list[int]:
        """Joint radians by name -> registers in slot order. Unverified axes: LEAVE."""
        out = [LEAVE] * N_SLOTS
        for axis in self.axes_of(side):
            value = q[axis.name]
            if not math.isfinite(value):
                raise HandMapError(f"non-finite {axis.name}: {value}")
            out[axis.slot] = axis.to_reg(value) if axis.verified else LEAVE
        return out

    def to_rad(self, registers: Sequence[float], *, side: str | None) -> dict[str, float]:
        if len(registers) != N_SLOTS:
            raise HandMapError(f"expected {N_SLOTS} registers, got {len(registers)}")
        return {a.name: a.to_rad(registers[a.slot]) for a in self.axes_of(side)}


def load_rh56f1_map(path: Path = DEFAULT_MAP) -> Rh56f1Map:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    slot_order = list(raw["slot_order"])
    joint_order = tuple(raw["joint_order"])
    if sorted(slot_order) != sorted(joint_order) or len(slot_order) != N_SLOTS:
        raise HandMapError("slot_order and joint_order must name the same 6 joints")
    axes = []
    for name in joint_order:
        j = raw["joints"][name]
        verified = j.get("verified", False) is True
        axes.append(Rh56f1Axis(
            name, slot_order.index(name),
            (float(j["rad"][0]), float(j["rad"][1])),
            (float(j["reg"][0]), float(j["reg"][1])), verified,
        ))
    side_axes = {}
    for side, cal in (raw.get("calibration") or {}).items():
        if side not in ("right", "left"):
            raise HandMapError(f"calibration side must be right or left: {side}")
        calibrated = []
        for axis in axes:
            c = (cal or {}).get(axis.name)
            if c is None:
                calibrated.append(axis)
                continue
            reg0, slope = float(c["reg0"]), float(c["deg_per_10reg"])
            if slope <= 0:
                raise HandMapError(f"calibration.{side}.{axis.name}: deg_per_10reg must be > 0")
            direction = 1 if axis.reg[1] > axis.reg[0] else -1
            end = axis.reg[1] + 50 * direction
            q_end = math.radians((reg0 - end) * slope / 10.0)
            calibrated.append(Rh56f1Axis(
                axis.name, axis.slot, (0.0, q_end), (reg0, end), axis.verified,
                cmd=(int(min(axis.reg)), int(max(axis.reg))),
            ))
        side_axes[side] = tuple(calibrated)
    return Rh56f1Map(joint_order, tuple(axes), side_axes)
