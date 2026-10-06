"""Position-based admittance per finger: the command is the operator's target minus a force term.

    q_cmd = q_operator - y,   y <- y + a * (f / K - y),   a = dt / (tau + dt)

f is the joint's actuator force over its rest reading (g, firmware mode 0 position control; minus a
small deadband). In free space f ~ 0 (after the force zero calibration) and the finger follows the
glove exactly; in contact the target backs off as force builds and settles where

    force ~ K * (how far the operator closes past the contact)

whatever the object and the touching link (10.06: a position offset alone gave 3-55 g per register,
docs/RH56F1_HAND_TUNING.md). Above max_force_g the term grows over_stiffness times faster, so a
far-closing operator meets a soft ceiling (~800 g + 200 g per rad more) instead of the 1.1-1.85 kg
saturation. The same law in one equation, no switch. No modes, no thresholds: the force feedback is always on and only acts
when there is force (user 10.06: a position + force loop runs together, not switched; the hand's
lead-screw linkage is non-backdrivable and stiff, so admittance, not impedance).

The low-pass on the force term keeps the loop stable on stiff contacts: per tick the loop gain is about
a * k_contact / K (k_contact up to ~29 kg/rad on thumb_2), so tau sets how fast a grip settles.
While only the actuator sees force and the fingertip sensor does not, the contact is on link 1, whose
shorter lever makes the real contact force larger than the reading (manual 2.5.12): the force is
counted 1 / proximal_scale times. Joints whose force sign is not known (thumb_1: reads negative under
an outward load) are left to the grip guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

TIP_OF = {"index_1": "index", "middle_1": "middle", "ring_1": "ring", "pinky_1": "pinky", "thumb_2": "thumb"}


@dataclass(frozen=True)
class AdmittanceConfig:
    enabled: bool = True
    joints: tuple[str, ...] = ("index_1", "middle_1", "ring_1", "pinky_1", "thumb_2")
    stiffness_g_per_rad: float = 2000.0  # K: grip force per rad the operator closes past the contact
    deadband_g: float = 40.0  # force readings below this (over rest) are not contact
    filter_tau_s: float = 0.5  # low-pass of the force term in contact (stability on stiff contacts)
    release_tau_s: float = 0.15  # ... once the force is gone (the offset fades so re-gripping is not slowed)
    rate_on_g: float = 60.0  # over the deadband: from this force on the command closes at a bounded rate (fast free-space
    #                          closing reads 50-60 g, 10.06)
    rate_rad_s: float = 0.3  # ... at most this x (1 - f / max_force), from the measured angle when the force came on
    max_force_g: float = 800.0  # above this the force term backs off far faster (operator closing a lot)
    over_stiffness_g_per_rad: float = 200.0  # ... at this stiffness: f = 800 + 200 g per rad more closing
    max_offset_rad: float = 1.6  # never more than the whole finger range
    proximal_scale: float = 1.0  # tip sensor quiet (link 1 contact): force / this; 10.06 user (A): 1.0, no boost
    tip_contact_n: float = 0.2

    def __post_init__(self) -> None:
        if (self.stiffness_g_per_rad <= 0 or self.over_stiffness_g_per_rad <= 0 or self.max_force_g <= 0
                or self.filter_tau_s < 0 or self.release_tau_s < 0 or self.deadband_g < 0 or self.max_offset_rad <= 0 or self.rate_on_g < 0 or self.rate_rad_s <= 0):
            raise ValueError("admittance: stiffness, over_stiffness, max_force, max_offset > 0; filter_tau, deadband >= 0")
        if not 0 < self.proximal_scale <= 1:
            raise ValueError("admittance: proximal_scale in (0, 1]")


def admittance_config(raw: Mapping | None) -> AdmittanceConfig:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(AdmittanceConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"admittance: unknown keys {unknown}")
    enabled = bool(raw.pop("enabled", True))
    joints = tuple(str(j) for j in raw.pop("joints", AdmittanceConfig.joints))
    bad = [j for j in joints if j not in TIP_OF]
    if bad:
        raise ValueError(f"admittance: joints {bad} have no known force sign (allowed: {sorted(TIP_OF)})")
    return AdmittanceConfig(enabled=enabled, joints=joints, **{k: float(v) for k, v in raw.items()})


@dataclass
class Admittance:
    config: AdmittanceConfig = field(default_factory=AdmittanceConfig)
    offset: dict[str, float] = field(default_factory=dict)  # y per joint (rad, >= 0: backs the command off)
    force: dict[str, float] = field(default_factory=dict)  # the force term used (g)
    _t: float | None = None
    _lim: dict[str, float] = field(default_factory=dict)  # rate-limited ceiling per joint while limiting
    _tc: float | None = None

    def reset(self) -> None:
        self.offset, self.force, self._t, self._lim, self._tc = {}, {}, None, {}, None

    def ceilings(self, t: float, q_measured: Mapping[str, float] | None,
                 q_last: Mapping[str, float] | None = None) -> dict[str, float]:
        """Once the force passes rate_on_g the command restarts from the measured angle and closes at most
        rate_rad_s x (1 - f / max_force) (opening at once: from `q_last`, the command last sent) until the force
        and the offset are gone, as the driver's admittance (robot_control rh56f1_hand). 10.06: a lead cap vs the
        measured angle held a rigid cup at ~350 g whatever the penetration (55 g per register past the finger)."""
        c = self.config
        dt = 0.0 if self._tc is None else min(max(t - self._tc, 0.0), 0.1)
        self._tc = t
        if not c.enabled or not q_measured:
            self._lim = {}
            return {}
        out = {}
        for j, f in self.force.items():
            if j not in q_measured:
                continue
            if j not in self._lim and f > c.rate_on_g:
                self._lim[j] = float(q_measured[j])
            if j in self._lim:
                base = min(self._lim[j], float(q_last[j])) if q_last and j in q_last else self._lim[j]
                out[j] = self._lim[j] = base + c.rate_rad_s * max(0.0, 1.0 - f / c.max_force_g) * dt
                if f <= 0 and self.offset.get(j, 0.0) < 0.002:
                    del self._lim[j]
        return out

    def update(self, t: float, force_rel: Mapping[str, float], tips_n: Mapping[str, float]) -> dict[str, float]:
        """{joint: rad to subtract from the operator target} for this tick."""
        c = self.config
        if not c.enabled:
            self.reset()
            return {}
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), 0.1)
        self._t = t
        out = {}
        for j in c.joints:
            f = max(float(force_rel.get(j, 0.0)) - c.deadband_g, 0.0)
            if f > 0 and tips_n.get(TIP_OF[j], 0.0) < c.tip_contact_n:
                f /= c.proximal_scale
            tau = c.filter_tau_s if f > 0 else c.release_tau_s
            a = 1.0 if tau == 0 else dt / (tau + dt)
            goal = f / c.stiffness_g_per_rad + max(f - c.max_force_g, 0.0) / c.over_stiffness_g_per_rad
            goal = min(goal, c.max_offset_rad)
            y = self.offset.get(j, 0.0)
            y += a * (goal - y)
            self.offset[j], self.force[j] = y, f
            if y > 1e-4:
                out[j] = y
        return out

    def record(self) -> dict | None:
        active = {j: {"offset_rad": round(y, 4), "force_g": round(self.force.get(j, 0.0))}
                  for j, y in self.offset.items() if y > 1e-4}
        return active or None


def stiff_contact(trace: list[tuple[float, float]], slope_g_per_rad: float = 8000.0, span_rad: float = 0.01,
                  min_span_rad: float = 0.004) -> float | None:
    """First measured angle from which the force climbs at least `slope_g_per_rad` (the object stops giving way).

    trace: (measured angle rad, force g over rest) while closing slowly, the angle increasing. The slope is taken
    over `span_rad` ahead, or up to the last sample when the trace ends sooner (a stall: the force rises, the angle
    does not), but never over less than `min_span_rad` (1 register ~0.0018 rad). 10.06 right index + cup: a first
    touch at 120 g, then 30-60 registers more at 5-7 g/register before the cup stopped giving (rigid: 55-100).
    May land up to `span_rad` early (the window straddles the knee): <= ~20 g of penetration at 2000 g/rad.
    None: never that stiff.
    """
    for i, (q0, f0) in enumerate(trace):
        ahead = [(q, f) for q, f in trace[i + 1:] if q - q0 >= span_rad]
        q1, f1 = ahead[0] if ahead else trace[-1]
        if q1 - q0 >= min_span_rad and (f1 - f0) / (q1 - q0) >= slope_g_per_rad:
            return q0
        if not ahead:
            return None
    return None
