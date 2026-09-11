"""Runway-relative geometry for the conventional landing autopilot.

These helpers are pure functions of position and the runway frame. They do
not move the aircraft; JSBSim remains the only integrator.
"""

from __future__ import annotations

import math

from fly_pilot.geodesy import heading_components
from fly_pilot.runway import Runway

# Aiming point past the threshold, metres along-track.
DEFAULT_AIM_ALONG_M = 120.0
# Matches the Milestone 1 short-final geometry (~250 m AGL at 3.2 km).
DEFAULT_GLIDESLOPE_DEG = 4.4


def wrap_degrees(angle_deg: float) -> float:
    """Wrap a heading/angle into (-180, 180]."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def heading_error_deg(heading_deg: float, target_deg: float) -> float:
    """Signed heading error: positive means the aircraft must turn right."""
    return wrap_degrees(target_deg - heading_deg)


def spawn_enu(
    runway: Runway,
    distance_m: float,
    right_m: float,
    agl_m: float,
) -> tuple[float, float, float]:
    """ENU of a final-approach spawn: `distance_m` before threshold, `right_m` right of centerline."""
    east_h, north_h = heading_components(runway.heading_deg)
    east = -distance_m * east_h + right_m * north_h
    north = -distance_m * north_h + right_m * (-east_h)
    return east, north, agl_m


def desired_glideslope_agl_m(
    along_m: float,
    *,
    aim_along_m: float = DEFAULT_AIM_ALONG_M,
    glideslope_deg: float = DEFAULT_GLIDESLOPE_DEG,
    min_range_m: float = 30.0,
) -> float:
    """Geometric AGL of a constant-angle glideslope to the aiming point.

    The slope goes to a small floor once the aircraft is at/past the aiming
    point so the flare law can take over rather than commanding a dive into
    the pavement.
    """
    range_m = max(aim_along_m - along_m, min_range_m)
    height = math.tan(math.radians(glideslope_deg)) * range_m
    if along_m >= aim_along_m:
        return min(height, 0.8)
    return height


def glideslope_error_m(
    alt_agl_m: float,
    along_m: float,
    *,
    aim_along_m: float = DEFAULT_AIM_ALONG_M,
    glideslope_deg: float = DEFAULT_GLIDESLOPE_DEG,
) -> float:
    """Height above the reference glideslope, metres. Positive = too high."""
    return float(alt_agl_m) - desired_glideslope_agl_m(
        along_m, aim_along_m=aim_along_m, glideslope_deg=glideslope_deg
    )


def glideslope_error_deg(
    alt_agl_m: float,
    along_m: float,
    *,
    aim_along_m: float = DEFAULT_AIM_ALONG_M,
    glideslope_deg: float = DEFAULT_GLIDESLOPE_DEG,
) -> float:
    """Angular error from the reference glideslope. Positive = above the slope."""
    range_m = max(aim_along_m - along_m, 1.0)
    actual_deg = math.degrees(math.atan2(max(alt_agl_m, 0.0), range_m))
    return actual_deg - glideslope_deg


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
