"""Per-arm gripper gesture for returning home and entering standby."""

from __future__ import annotations

GRIPPER_PARK_HOLD_S = 2.0
GRIPPER_FULLY_CLOSED = 0.005
GRIPPER_REOPENED = 0.15


class GripperHomeStandby:
    """Detect per-side close-to-park and reopen-to-wake transitions."""

    def __init__(
        self,
        *,
        hold_s: float = GRIPPER_PARK_HOLD_S,
        closed_threshold: float = GRIPPER_FULLY_CLOSED,
        reopened_threshold: float = GRIPPER_REOPENED,
        initial_standby: bool = False,
    ) -> None:
        if hold_s < 0.0:
            raise ValueError("hold_s must be >= 0")
        if not 0.0 <= closed_threshold < reopened_threshold <= 1.0:
            raise ValueError(
                "gripper thresholds must satisfy 0 <= closed < reopened <= 1"
            )
        self._hold_s = float(hold_s)
        self._closed_threshold = float(closed_threshold)
        self._reopened_threshold = float(reopened_threshold)
        self._closed_since: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._standby = {
            "left": bool(initial_standby),
            "right": bool(initial_standby),
        }

    def update(
        self,
        openings: dict[str, float],
        now_s: float,
        enabled_sides: tuple[str, ...],
        *,
        wake_openings: dict[str, float] | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(park_sides, wake_sides)`` for this input sample.

        ``openings`` is measured physical robot feedback used by the
        close-to-park timer. ``wake_openings`` may come from the operator
        controller so a parked arm can be reactivated while its robot gripper
        remains closed.
        """
        park: list[str] = []
        wake: list[str] = []
        for side in enabled_sides:
            if self._standby[side]:
                wake_source = (
                    wake_openings if wake_openings is not None else openings
                )
                if side not in wake_source:
                    continue
                wake_opening = float(wake_source[side])
                if wake_opening >= self._reopened_threshold:
                    self._standby[side] = False
                    self._closed_since[side] = None
                    wake.append(side)
                continue

            if side not in openings:
                self._closed_since[side] = None
                continue
            opening = float(openings[side])
            if opening > self._closed_threshold:
                self._closed_since[side] = None
                continue
            closed_since = self._closed_since[side]
            if closed_since is None:
                self._closed_since[side] = float(now_s)
            elif now_s - closed_since >= self._hold_s:
                self._standby[side] = True
                self._closed_since[side] = None
                park.append(side)
        return tuple(park), tuple(wake)

    def is_standby(self, side: str) -> bool:
        return self._standby[side]

    def standby_sides(self, enabled_sides: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(side for side in enabled_sides if self._standby[side])

    def enter_standby(self, sides: tuple[str, ...]) -> None:
        """Require the selected arms to be reopened before they can wake."""
        for side in sides:
            self._standby[side] = True
            self._closed_since[side] = None

    def reset(self) -> None:
        for side in self._closed_since:
            self._closed_since[side] = None
            self._standby[side] = False

    def reset_close_timers(self) -> None:
        """Cancel pending close-to-park timers without changing arm state."""
        for side in self._closed_since:
            self._closed_since[side] = None
