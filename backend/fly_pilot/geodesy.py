"""Local ENU conversions around a runway origin.

Uses a first-order WGS84 metres-per-degree approximation. Accurate enough
for a few kilometres of visual and episode geometry; JSBSim remains the
authoritative geodetic propagator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Return (metres per deg latitude, metres per deg longitude) at lat."""
    lat = math.radians(lat_deg)
    m_lat = (
        111132.92
        - 559.82 * math.cos(2.0 * lat)
        + 1.175 * math.cos(4.0 * lat)
    )
    m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3.0 * lat)
    return m_lat, m_lon


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """East-North-Up frame anchored at a geodetic origin."""

    lat0_deg: float
    lon0_deg: float
    alt0_m: float = 0.0

    def to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
        m_lat, m_lon = meters_per_degree(self.lat0_deg)
        east = (lon_deg - self.lon0_deg) * m_lon
        north = (lat_deg - self.lat0_deg) * m_lat
        up = alt_m - self.alt0_m
        return east, north, up

    def from_enu(self, east_m: float, north_m: float, up_m: float) -> tuple[float, float, float]:
        m_lat, m_lon = meters_per_degree(self.lat0_deg)
        lat = self.lat0_deg + north_m / m_lat
        lon = self.lon0_deg + east_m / m_lon
        alt = self.alt0_m + up_m
        return lat, lon, alt


def heading_components(heading_deg: float) -> tuple[float, float]:
    """Unit vector (east, north) of a true heading in degrees."""
    heading = math.radians(heading_deg)
    return math.sin(heading), math.cos(heading)


def runway_coordinates(
    east_m: float, north_m: float, heading_deg: float
) -> tuple[float, float]:
    """Project ENU into runway along-track / right-of-track metres.

    Along-track is positive in the landing direction from the threshold.
    Right-track is positive to the right of the landing direction.
    """
    heading = math.radians(heading_deg)
    s = math.sin(heading)
    c = math.cos(heading)
    along = east_m * s + north_m * c
    right = east_m * c - north_m * s
    return along, right
