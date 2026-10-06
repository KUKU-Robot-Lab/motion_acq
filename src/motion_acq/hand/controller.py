"""Hand enable/fault state machine around HandRetargeter, free of ROS.

The ROS hand_node feeds it glove samples, driver feedback and enable
requests, calls tick() at the control rate and publishes what it returns.

States
    DISABLED  nothing is published.
    ENABLED   streaming angle_set. Start and end pose is home (hand open):
              after enable the hand walks home (HOMING), takes the operator's reference
              pose (REFERENCE, see below) and only then follows
              the glove (RUNNING, or HOLD with nothing published while the
              glove is stale or frozen); a disable walks it home again and
              then disables. Not reaching home in home_timeout_s is a FAULT
              at the start and a plain disable at the end.
    FAULT     latched; nothing is published until a disable and a new enable.

Enable needs: a fresh angle_actual whose six registers are plausible (no 0,
-1 or 65535 sentinels, within each axis' command range ± margin), and every
driver topic subscribed. The hand then starts from that measured pose;
speed/force are sent at enable and re-sent every resend_s. Losing
angle_actual for longer than measured_stale_s while enabled is a FAULT.
A FAULT does not move the hand (its feedback cannot be trusted).

Reference (10.06 user: take the current pose as the reference before linking the glove):
at home the operator holds the reference pose (configs reference.pose, the robot's home:
fingers straight, thumb beside the index) still for reference_s; that take re-zeroes the
operator's saved map and re-aligns the glove hand model (calibration.rezero), then the
hand follows. Not still within reference_timeout_s: it follows the saved calibration and
says so.

Frozen hand: 10.05 22:24 the left RH56F1 dropped off EtherCAT during a firm cup grasp
(master: "WKC -1 < 3 for 100 cycles") and the driver kept publishing its last state on
time: angles, joint forces and tip forces bit-identical for 24 s. The hand ignored the
open command, and the stale contact kept the glove braked. Live RH56F1 data changes at
least every ~70 ms (forces are never exactly still), so with the force or touch stream
present, a state with nothing changed for hand_frozen_s while enabled is a FAULT; the
glove feedback goes off with it.

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

from motion_acq.hand.grip_guard import GripGuard
from motion_acq.hand.feedback import OFF, HapticFeedback, Haptics
from motion_acq.hand.nova2 import SIDE_PREFIX, glove_joint_names
from motion_acq.hand.retarget import HandRetargeter, HandState
from motion_acq.hand.rh56f1 import N_SLOTS, Rh56f1Map

MEASURED_MARGIN = 100  # registers beyond the command range still accepted as plausible
SENTINELS = (-1, 0, 65535)


class Mode(str, Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    FAULT = "fault"


class Phase(str, Enum):
    TO_HOME = "to_home"  # after enable: walk home before following
    REFERENCE = "reference"  # at home: the operator holds the reference pose still (re-zero, align)
    FOLLOW = "follow"
    RETURN = "return_home"  # after disable: walk home, then disable


@dataclass(frozen=True)
class ControllerConfig:
    glove_stale_s: float = 0.2
    glove_frozen_s: float = 1.0
    measured_stale_s: float = 0.5
    hand_frozen_s: float = 0.5
    resend_s: float = 1.0
    home_timeout_s: float = 5.0
    home_tolerance_registers: int = 30  # ~3 deg on the fingers
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
    haptics: Haptics = OFF  # Nova 2 feedback for this tick (all off unless following the glove)
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
        self.phase = Phase.TO_HOME
        self._phase_t0: float | None = None
        self.last_refusal: str | None = None
        self.home_rad = dict(retargeter.config.home_rad)
        self._home_registers = hand_map.to_registers(self.home_rad, side=side)
        self.feedback = HapticFeedback(retargeter.config.feedback, retargeter.config.closed_rad)
        self.grip_guard = GripGuard(retargeter.config.grip_guard)
        self._last_reading: dict[str, object] = {}
        self._reference: list[tuple[float, dict]] = []  # (t, glove signals) while in REFERENCE
        self.reference_note: str | None = None
        self._status: dict | None = None
        self._hand_changed_t = -math.inf
        self._sensor_streams = False

    # -- inputs -----------------------------------------------------------
    def on_glove(self, angles: Mapping[str, float], t: float) -> None:
        if self.glove is None or dict(angles) != dict(self.glove[0]):
            self._glove_changed_t = t
        self.glove = (angles, t)

    def glove_frozen(self, t: float) -> bool:
        return self.glove is not None and t - self._glove_changed_t > self.config.glove_frozen_s

    def on_touch(self, finger_forces, palm_data, t: float) -> None:
        self._hand_changed("touch", (tuple(finger_forces), tuple(palm_data)), t)
        self.feedback.on_touch(finger_forces, palm_data, t)

    def on_joint_force(self, names, values, t: float) -> None:
        self._hand_changed("force", tuple(values), t)
        self.feedback.on_joint_force(names, values, t)
        self.grip_guard.on_force(names, values, t)

    def on_current(self, names, values, t: float) -> None:
        """GetCurrentAct1 (mA per actuator): the grip guard's second reading."""
        self.grip_guard.on_current(names, values, t)

    def _hand_changed(self, stream: str, value, t: float) -> None:
        """Remember when any RH56F1 reading last changed (frozen-hand check)."""
        if stream != "angle":
            self._sensor_streams = True
        if self._last_reading.get(stream) != value:
            self._last_reading[stream] = value
            self._hand_changed_t = t

    def hand_frozen(self, t: float) -> bool:
        return self._sensor_streams and t - self._hand_changed_t > self.config.hand_frozen_s

    def on_status(self, status: dict) -> None:
        """The driver's 1 Hz status (state codes, errors, WKC): logged on the next tick."""
        self._status = status

    def on_glove_error(self) -> None:
        self.glove_errors += 1

    def on_measured(self, registers: Sequence[int], hand_id: int, t: float) -> None:
        self.measured = ([int(v) for v in registers], int(hand_id), t)
        self._hand_changed("angle", tuple(self.measured[0]), t)

    def request_enable(self, enable: bool) -> None:
        self.want_enable = bool(enable)
        if self.mode is Mode.ENABLED:
            if enable and self.phase is Phase.RETURN:
                self._set_phase(Phase.TO_HOME, None)
            elif not enable and self.phase is not Phase.RETURN:
                self._set_phase(Phase.RETURN, None)
            return
        if not enable and self.mode is Mode.FAULT:
            self.mode = Mode.DISABLED
            self.fault_reason = None
            self.retargeter.state = HandState.IDLE

    def _set_phase(self, phase: Phase, t: float | None) -> None:
        """t None: the phase clock starts at the next tick (request_enable has no time)."""
        self.phase = phase
        self._phase_t0 = t

    @property
    def busy(self) -> bool:
        """Still moving to its end pose (the node keeps spinning until False)."""
        return self.mode is Mode.ENABLED and self.phase is Phase.RETURN

    def _at_home(self, t: float) -> bool:
        command = self.retargeter.command()
        if any(abs(command[j] - q) > 1e-3 for j, q in self.home_rad.items()):
            return False
        if not self._measured_fresh(t):
            return False
        assert self.measured is not None
        tol = self.config.home_tolerance_registers
        return all(abs(m - h) <= tol for m, h in zip(self.measured[0], self._home_registers, strict=True))

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
        if self.hand_frozen(t):
            return "RH56F1 readings frozen (EtherCAT link or hand power?): restart the hand driver"
        assert self.measured is not None
        reasons = implausible_registers(self.measured[0], self.hand_map, self.side)
        if reasons:
            return "implausible angle_actual: " + "; ".join(reasons)
        self.retargeter.start(self.hand_map.to_rad(self.measured[0], side=self.side), t)
        self.mode = Mode.ENABLED
        self._set_phase(Phase.TO_HOME, t)
        self._last_resend = -math.inf
        return None

    def _step_enabled(self, t: float):
        """One ENABLED cycle: walk home, follow the glove, or walk home and disable."""
        if self.phase is Phase.FOLLOW:
            glove = self.glove
            fresh = (glove is not None and t - glove[1] <= self.config.glove_stale_s
                     and not self.glove_frozen(t))
            ceilings = self.grip_guard.ceilings(self._measured_rad() if self._measured_fresh(t) else None, t)
            return self.retargeter.step(glove[0] if fresh and glove else None, t, ceilings)
        if self._phase_t0 is None:
            self._phase_t0 = t
        step = self.retargeter.step_to(self.home_rad, t)
        if self.phase is Phase.REFERENCE:
            self._take_reference(t)
            return step
        if self._at_home(t):
            if self.phase is Phase.TO_HOME:
                cfg = self.retargeter.config
                self._reference, self.reference_note = [], None
                self._set_phase(Phase.REFERENCE if cfg.reference_s > 0 else Phase.FOLLOW, t)
            else:
                self.mode = Mode.DISABLED
                self.retargeter.state = HandState.IDLE
        elif t - self._phase_t0 > self.config.home_timeout_s:
            reason = f"home not reached in {self.config.home_timeout_s:g} s"
            if self.phase is Phase.TO_HOME:
                self.mode = Mode.FAULT
                self.fault_reason = reason
            else:
                self.mode = Mode.DISABLED
                self.last_refusal = reason
            self.retargeter.state = HandState.IDLE
        return step

    def _take_reference(self, t: float) -> None:
        """Collect the operator's still reference pose; then re-zero and align, and follow."""
        from motion_acq.hand.calibration import CalibrationError, rezero, unsteady_inputs

        cfg = self.retargeter.config
        glove = self.glove
        if glove is not None and t - glove[1] <= self.config.glove_stale_s and not self.glove_frozen(t):
            if not self._reference or self._reference[-1][1] is not glove[0]:
                self._reference.append((t, dict(glove[0])))
        self._reference = [(ts, g) for ts, g in self._reference if t - ts <= cfg.reference_s]
        window = [g for _, g in self._reference]
        cal = self.retargeter.calibration
        covered = bool(self._reference) and t - self._reference[0][0] >= 0.9 * cfg.reference_s
        if covered and len(window) >= 10 and not unsteady_inputs(window, cal.inputs):
            try:
                self.retargeter.set_calibration(rezero(cal, window, cfg.reference_pose))
                aligned = "aligned" if self.retargeter.calibration.alignment is not None else "no hand model"
                self.reference_note = f"reference taken ({len(window)} samples, {aligned})"
            except CalibrationError as exc:
                self.reference_note = f"reference failed ({exc}): following the saved calibration"
            self._set_phase(Phase.FOLLOW, t)
        elif t - (self._phase_t0 or t) > cfg.reference_timeout_s:
            self.reference_note = (f"reference pose not held still in {cfg.reference_timeout_s:g} s: "
                                   "following the saved calibration")
            self._set_phase(Phase.FOLLOW, t)

    def tick(self, t: float, *, subscribers_ready: bool) -> Outputs:
        out = Outputs(hand_id=None if self.measured is None else self.measured[1])
        if self.want_enable and self.mode is Mode.DISABLED:
            self.last_refusal = self._try_enable(t, subscribers_ready)
        if self.mode is Mode.ENABLED and not self._measured_fresh(t):
            self.mode = Mode.FAULT
            self.fault_reason = f"angle_actual lost for > {self.config.measured_stale_s} s"
            self.retargeter.state = HandState.IDLE
        if self.mode is Mode.ENABLED and self.hand_frozen(t):
            self.mode = Mode.FAULT
            self.fault_reason = (f"RH56F1 readings unchanged for > {self.config.hand_frozen_s} s: the hand stopped "
                                 "answering (EtherCAT link or hand power?) / 손 응답 없음: 드라이버 로그의 WKC 확인")
            self.retargeter.state = HandState.IDLE
        step = None
        if self.mode is Mode.ENABLED:
            if t - self._last_resend >= self.config.resend_s:
                self._last_resend = t
                out.speed = [self.config.driver_speed] * N_SLOTS
                out.force = [self.config.driver_force] * N_SLOTS
            step = self._step_enabled(t)
            if step.state in (HandState.RUNNING, HandState.HOMING) and step.registers is not None:
                out.angle = list(step.registers)
        following = (self.mode is Mode.ENABLED and self.phase is Phase.FOLLOW
                      and step is not None and step.state is HandState.RUNNING)
        out.haptics = self.feedback.update(t, active=following,
                                           q_command=step.q_command if step else None,
                                           q_measured=self._measured_rad())
        sensors = self.feedback.sensors(t)
        glove_age = None if self.glove is None else round(t - self.glove[1], 4)
        out.record = {
            "t_mono_s": round(t, 6),
            "side": self.side,
            "mode": self.mode.value,
            "phase": self.phase.value if self.mode is Mode.ENABLED else None,
            "state": step.state.value if step else HandState.IDLE.value,
            "fault": self.fault_reason,
            "refusal": self.last_refusal if self.mode is Mode.DISABLED else None,
            "glove_age_s": glove_age,
            "glove_frozen": self.glove_frozen(t),
            "glove_errors": self.glove_errors,
            "features": step.features if step else None,
            "normalized": step.normalized if step else None,
            "q_target_rad": step.q_target if step else None,
            "missing_inputs": self.retargeter.missing_inputs or None,
            "method": self.retargeter.method_used if following else None,
            "reference": self.reference_note,
            "q_command_rad": step.q_command if step else None,
            "registers": out.angle,
            "measured_registers": None if self.measured is None else self.measured[0],
            "measured_rad": self._measured_rad(),
            "glove_angles": self._glove_vector(),
            "hand_id": out.hand_id,
            **sensors,
            "haptics": out.haptics.to_dict(),
            "grip_guard": dict(self.grip_guard.engaged) if following and self.grip_guard.engaged else None,
            "current_ma": self.grip_guard.current_now(t),
            "hand_status": self._status,  # driver status, only on the tick after one arrived (1 Hz)
        }
        self._status = None
        return out
