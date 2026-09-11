from fly_pilot.guidance import (
    desired_glideslope_agl_m,
    glideslope_error_deg,
    glideslope_error_m,
    heading_error_deg,
    spawn_enu,
    wrap_degrees,
)
from fly_pilot.runway import Runway


def test_wrap_degrees() -> None:
    assert wrap_degrees(90) == 90
    assert abs(wrap_degrees(270) - (-90)) < 1e-9
    assert abs(wrap_degrees(-190) - 170) < 1e-9
    assert abs(heading_error_deg(350, 10) - 20) < 1e-9
    assert abs(heading_error_deg(10, 350) + 20) < 1e-9


def test_spawn_enu_right_of_runway_09_is_south() -> None:
    east, north, up = spawn_enu(Runway(heading_deg=90.0), distance_m=3200.0, right_m=50.0, agl_m=250.0)
    assert abs(east + 3200.0) < 1e-6
    assert abs(north + 50.0) < 1e-6
    assert up == 250.0


def test_desired_glideslope_matches_nominal_approach_geometry() -> None:
    # 3200 m before threshold, 4.4° to an aiming point 120 m past threshold.
    agl = desired_glideslope_agl_m(-3200.0, aim_along_m=120.0, glideslope_deg=4.4)
    assert 250.0 < agl < 262.0
    # Above the slope at the default 250 m spawn.
    err_m = glideslope_error_m(250.0, -3200.0, aim_along_m=120.0, glideslope_deg=4.4)
    assert -12.0 < err_m < 0.0
    err_deg = glideslope_error_deg(250.0, -3200.0, aim_along_m=120.0, glideslope_deg=4.4)
    assert abs(err_deg) < 0.3


def test_glideslope_error_sign_is_above_positive() -> None:
    high = glideslope_error_m(300.0, -2000.0)
    low = glideslope_error_m(50.0, -2000.0)
    assert high > 0.0
    assert low < 0.0
    assert glideslope_error_deg(300.0, -2000.0) > 0.0
