"""Shared observation and control types.

These are the only values controllers are allowed to consume and produce.
Keeping them independent of JSBSim and WebSockets lets ManualController,
and later MaleCNS / expert controllers, share one interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EpisodeStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    LANDED = "landed"
    CRASHED = "crashed"
    OUT_OF_BOUNDS = "out_of_bounds"
    FAILED_APPROACH = "failed_approach"


@dataclass(frozen=True, slots=True)
class AircraftControls:
    """Normalised aircraft inceptors.

    aileron:  [-1, 1]  positive rolls right
    elevator: [-1, 1]  positive is stick-back / nose-up (pilot convention)
    rudder:   [-1, 1]  positive yaws right
    throttle: [0, 1]   0 idle, 1 full
    """

    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.5

    def clamped(self) -> AircraftControls:
        return AircraftControls(
            aileron=_clamp(self.aileron, -1.0, 1.0),
            elevator=_clamp(self.elevator, -1.0, 1.0),
            rudder=_clamp(self.rudder, -1.0, 1.0),
            throttle=_clamp(self.throttle, 0.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class AircraftObservation:
    """Authoritative aircraft/environment snapshot derived from JSBSim."""

    sim_time_s: float
    lat_deg: float
    lon_deg: float
    alt_msl_m: float
    alt_agl_m: float
    east_m: float
    north_m: float
    up_m: float
    along_m: float
    right_m: float
    airspeed_kts: float
    groundspeed_kts: float
    vertical_speed_fpm: float
    pitch_deg: float
    roll_deg: float
    heading_deg: float
    alpha_deg: float
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    on_ground: bool
    wow: tuple[bool, bool, bool] = (False, False, False)
    p_deg_s: float = 0.0
    q_deg_s: float = 0.0
    r_deg_s: float = 0.0
    beta_deg: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeInfo:
    status: EpisodeStatus = EpisodeStatus.IN_PROGRESS
    reason: str | None = None
    touchdown_fpm: float | None = None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
