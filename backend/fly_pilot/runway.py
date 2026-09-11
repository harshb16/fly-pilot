"""Runway geometry and approach initial conditions."""

from __future__ import annotations

from dataclasses import dataclass

from fly_pilot.geodesy import LocalFrame, heading_components


@dataclass(frozen=True, slots=True)
class Runway:
    """A flat sea-level runway used as the visual and episode origin.

    The threshold is the geodetic origin. Landing direction is `heading_deg`.
    """

    name: str = "FP 09"
    lat_deg: float = 37.0
    lon_deg: float = -122.0
    alt_m: float = 0.0
    heading_deg: float = 90.0
    length_m: float = 1200.0
    width_m: float = 30.0

    @property
    def frame(self) -> LocalFrame:
        return LocalFrame(self.lat_deg, self.lon_deg, self.alt_m)


@dataclass(frozen=True, slots=True)
class ApproachConfig:
    """Final-approach spawn used at episode reset.

    Defaults sit in the requested 2–4 km / 200–400 m AGL window at a
    typical C172 short-final speed.
    """

    distance_m: float = 3200.0
    agl_m: float = 250.0
    airspeed_kts: float = 70.0
    flight_path_deg: float = -4.5
    alpha_deg: float = 4.0
    throttle: float = 0.5
    flaps_norm: float = 0.4
    mixture_norm: float = 0.9


def approach_origin(runway: Runway, config: ApproachConfig) -> tuple[float, float, float]:
    """Geodetic (lat, lon, alt_m) of the approach spawn."""
    east_h, north_h = heading_components(runway.heading_deg)
    east = -config.distance_m * east_h
    north = -config.distance_m * north_h
    return runway.frame.from_enu(east, north, config.agl_m)
