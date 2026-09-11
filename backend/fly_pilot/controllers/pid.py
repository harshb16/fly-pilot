"""Small PID helper for the conventional ExpertLandingController.

This is ordinary classical control. It is not part of MaleCNS.
"""

from __future__ import annotations

from fly_pilot.guidance import clamp


class PID:
    """Discrete PID with a clamped integrator (anti-windup)."""

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        lo: float = -1.0,
        hi: float = 1.0,
        i_limit: float = 0.4,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lo = lo
        self.hi = hi
        self.i_limit = i_limit
        self.integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float, deriv: float | None = None) -> float:
        dt = max(float(dt), 1e-4)
        self.integral = clamp(self.integral + error * dt, -self.i_limit, self.i_limit)
        if deriv is None:
            if self._prev_error is None:
                deriv = 0.0
            else:
                deriv = (error - self._prev_error) / dt
        self._prev_error = error
        return clamp(self.kp * error + self.ki * self.integral + self.kd * deriv, self.lo, self.hi)
