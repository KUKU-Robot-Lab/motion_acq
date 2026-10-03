"""Scalar signal shaping shared by the head and hand retargeters.

Dependency-free apart from numpy, so ROS nodes running on the system Python
(3.10 on Humble) can import it from the motion_acq source tree.
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) for one scalar signal."""

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        if min_cutoff_hz <= 0.0 or d_cutoff_hz <= 0.0 or beta < 0.0:
            raise ValueError("One Euro needs positive cutoffs and beta >= 0.")
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff_hz = float(d_cutoff_hz)
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, value: float | None = None, t_s: float | None = None) -> None:
        self._x = None if value is None else float(value)
        self._dx = 0.0
        self._t = t_s

    def __call__(self, value: float, t_s: float) -> float:
        value = float(value)
        if self._x is None or self._t is None:
            self._x, self._t = value, t_s
            return value
        dt = t_s - self._t
        if dt <= 0.0:
            return self._x
        dx = (value - self._x) / dt
        self._dx += self._alpha(self.d_cutoff_hz, dt) * (dx - self._dx)
        cutoff = self.min_cutoff_hz + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (value - self._x)
        self._t = t_s
        return self._x


class Deadband:
    """Backlash deadband: the output only moves once the input leaves ±width."""

    def __init__(self, width: float) -> None:
        if width < 0.0:
            raise ValueError("Deadband width must be >= 0.")
        self.width = float(width)
        self._out: float | None = None

    def reset(self, value: float | None = None) -> None:
        self._out = None if value is None else float(value)

    def __call__(self, value: float) -> float:
        value = float(value)
        if self._out is None:
            self._out = value
        elif value > self._out + self.width:
            self._out = value - self.width
        elif value < self._out - self.width:
            self._out = value + self.width
        return self._out


class RateLimiter:
    """Second-order follower: bounded velocity and acceleration, no overshoot."""

    def __init__(self, max_velocity: float, max_acceleration: float) -> None:
        if max_velocity <= 0.0 or max_acceleration <= 0.0:
            raise ValueError("Rate limits must be positive.")
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.position = 0.0
        self.velocity = 0.0

    def reset(self, position: float) -> None:
        self.position = float(position)
        self.velocity = 0.0

    def stop(self) -> None:
        self.velocity = 0.0

    def __call__(self, target: float, dt: float) -> float:
        if dt <= 0.0:
            return self.position
        error = float(target) - self.position
        # Fastest speed that can still brake to zero at the target when the
        # deceleration is applied in steps of dt (discrete braking curve).
        half = 0.5 * self.max_acceleration * dt
        braking = math.sqrt(half * half + 2.0 * self.max_acceleration * abs(error)) - half
        desired = math.copysign(min(self.max_velocity, braking), error)
        step = self.max_acceleration * dt
        self.velocity += float(np.clip(desired - self.velocity, -step, step))
        move = self.velocity * dt
        if abs(move) >= abs(error) and error * move >= 0.0:
            self.position = float(target)
            self.velocity = 0.0
        else:
            self.position += move
        return self.position
