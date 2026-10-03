"""Hand enable/fault state machine around HandRetargeter, free of ROS.

The ROS hand_node feeds it glove samples, driver feedback and enable
requests, calls tick() at the control rate and publishes what it returns.

States
    DISABLED  nothing is published.
    ENABLED   streaming angle_set; RUNNING while the glove is fresh, HOLD
              (nothing published) while it is stale or frozen.
    FAULT     latched; nothing is published until a disable and a new enable.

Enable needs: a fresh angle_actual whose six registers are plausible (no 0,
-1 or 65535 sentinels, within each axis' command range ± margin), and every
driver topic subscribed. The hand then starts from that measured pose;
speed/force are sent at enable and re-sent every resend_s. Losing
angle_actual for longer than measured_stale_s while enabled is a FAULT.
Disable freezes the hand: the last measured registers are sent once.

Frozen glove: when SenseCom dies, senseglove_ros keeps publishing its last
values on time and without error (seen on bumsu's setup, 2026-09-22). A glove
whose 20 angles have not changed at all for glove_frozen_s is therefore
treated as stale; any change resumes it. A worn glove never repeats exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from motion_acq.hand.nova2 import SIDE_PREFIX, glove_joint_names
from motion_acq.hand.retarget import HandRetargeter, HandState
from motion_acq.hand.rh56f1 import N_SLOTS, Rh56f1Map

MEASURED_MARGIN = 100  # registers beyond the command range still accepted as plausible
SENTINELS = (-1, 0, 65535)


class Mode(str, Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    FAULT = "fault"


@dataclass(frozen=True)
class ControllerConfig:
    glove_stale_s: float = 0.2
    glove_frozen_s: float = 1.0
    measured_stale_s: float = 0.5
    resend_s: float = 1.0
    driver_speed: int = 2000
    driver_force: int = 600

    def __post_init__(self) -> None:
        if not 0 < self.driver_speed <= 4000:
            raise ValueError(f"driver_speed must be in (0, 4000], not {self.driver_speed}")
        if not 0 < self.driver_force <= 1000:
            raise ValueError(f"driver_force must be in (0, 1000] g, not {self.driver_force}")


@dataclass
class Outputs:
    hand_id: int | None = None
    angle: list[int] | None = None
    speed: list[int] | None = None
    force: list[int] | None = None
    record: dict = field(default_factory=dict)


def implausible_registers(registers: Sequence[int], hand_map: Rh56f1Map, side: str) -> list[str]:
    """Reasons a driver feedback vector cannot seed the start pose (empty = plausible)."""
    if len(registers) != N_SLOTS:
        return [f"{len(registers)} registers, expected {N_SLOTS}"]
    reasons = []
    for axis in hand_map.axes_of(side):
        value = int(registers[axis.slot])
        lo, hi = axis.command_range
        if value in SENTINELS or not lo - MEASURED_MARGIN <= value <= hi + MEASURED_MARGIN:
            reasons.append(f"{axis.name} (slot {axis.slot}) = {value}, expected {lo}..{hi}")
    return reasons


class HandController:
    def __init__(
        self,
        retargeter: HandRetargeter,
        hand_map: Rh56f1Map,
        side: str,
        config: ControllerConfig,
    ) -> None:
        self.retargeter = retargeter
        self.hand_map = hand_map
        self.side = side
        self.config = config
        self.mode = Mode.DISABLED
        self.fault_reason: str | None = None
        self.want_enable = False
        self.glove: tuple[Mapping[str, float], float] | None = None
        self._glove_changed_t = -math.inf  # last arrival whose angles differed from the previous
        self.glove_errors = 0
        self.measured: tuple[list[int], int, float] | None = None  # registers, hand_id, t
        self._last_resend = -math.inf
        self._freeze_pending = False
        self.last_refusal: str | None = None

    # -- inputs -----------------------------------------------------------
    def on_glove(self, angles: Mapping[str, float], t: float) -> None:
        if self.glove is None or dict(angles) != dict(self.glove[0]):
            self._glove_changed_t = t
        self.glove = (angles, t)

    def glove_frozen(self, t: float) -> bool:
        return self.glove is not None and t - self._glove_changed_t > self.config.glove_frozen_s

    def on_glove_error(self) -> None:
        self.glove_errors += 1

    def on_measured(self, registers: Sequence[int], hand_id: int, t: float) -> None:
        self.measured = ([int(v) for v in registers], int(hand_id), t)

    def request_enable(self, enable: bool) -> None:
        self.want_enable = bool(enable)
        if enable:
            return
        if self.mode is Mode.ENABLED:
            self._freeze_pending = True
        if self.mode is not Mode.DISABLED:
            self.mode = Mode.DISABLED
            self.fault_reason = None
            self.retargeter.state = HandState.IDLE

    def _measured_rad(self) -> dict[str, float] | None:
        if self.measured is None or implausible_registers(self.measured[0], self.hand_map, self.side):
            return None
        return self.hand_map.to_rad(self.measured[0], side=self.side)

    def _glove_vector(self) -> list[float] | None:
        """Raw glove angles in glove_joint_names(side) order (dataset layout)."""
        if self.glove is None:
            return None
        angles = self.glove[0]
        prefix = len(SIDE_PREFIX[self.side]) + 1
        try:
            return [float(angles[name[prefix:]]) for name in glove_joint_names(self.side)]
        except KeyError:
            return None

    # -- control ----------------------------------------------------------
    def _measured_fresh(self, t: float) -> bool:
        return self.measured is not None and t - self.measured[2] <= self.config.measured_stale_s

    def _try_enable(self, t: float, subscribers_ready: bool) -> str | None:
        if not subscribers_ready:
            return "driver topics have no subscriber"
        if not self._measured_fresh(t):
            return "no fresh angle_actual"
        assert self.measured is not None
        reasons = implausible_registers(self.measured[0], self.hand_map, self.side)
        if reasons:
            return "implausible angle_actual: " + "; ".join(reasons)
        self.retargeter.start(self.hand_map.to_rad(self.measured[0], side=self.side), t)
        self.mode = Mode.ENABLED
        self._last_resend = -math.inf
        return None

    def tick(self, t: float, *, subscribers_ready: bool) -> Outputs:
        out = Outputs(hand_id=None if self.measured is None else self.measured[1])
        if self._freeze_pending:
            self._freeze_pending = False
            if self._measured_fresh(t):
                assert self.measured is not None
                out.angle = list(self.measured[0])  # stop where the hand is
        if self.want_enable and self.mode is Mode.DISABLED:
            self.last_refusal = self._try_enable(t, subscribers_ready)
        if self.mode is Mode.ENABLED and not self._measured_fresh(t):
            self.mode = Mode.FAULT
            self.fault_reason = f"angle_actual lost for > {self.config.measured_stale_s} s"
            self.retargeter.state = HandState.IDLE
        step = None
        if self.mode is Mode.ENABLED:
            if t - self._last_resend >= self.config.resend_s:
                self._last_resend = t
                out.speed = [self.config.driver_speed] * N_SLOTS
                out.force = [self.config.driver_force] * N_SLOTS
            glove = self.glove
            fresh = (glove is not None and t - glove[1] <= self.config.glove_stale_s
                     and not self.glove_frozen(t))
            step = self.retargeter.step(glove[0] if fresh and glove else None, t)
            if step.state is HandState.RUNNING and step.registers is not None:
                out.angle = list(step.registers)
        glove_age = None if self.glove is None else round(t - self.glove[1], 4)
        out.record = {
            "t_mono_s": round(t, 6),
            "side": self.side,
            "mode": self.mode.value,
            "state": step.state.value if step else HandState.IDLE.value,
            "fault": self.fault_reason,
            "refusal": self.last_refusal if self.want_enable and self.mode is Mode.DISABLED else None,
            "glove_age_s": glove_age,
            "glove_frozen": self.glove_frozen(t),
            "glove_errors": self.glove_errors,
            "features": step.features if step else None,
            "normalized": step.normalized if step else None,
            "q_target_rad": step.q_target if step else None,
            "q_command_rad": step.q_command if step else None,
            "registers": out.angle,
            "measured_registers": None if self.measured is None else self.measured[0],
            "measured_rad": self._measured_rad(),
            "glove_angles": self._glove_vector(),
            "hand_id": out.hand_id,
        }
        return out
