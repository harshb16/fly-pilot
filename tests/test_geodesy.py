from fly_pilot.geodesy import LocalFrame, heading_components, meters_per_degree, runway_coordinates
from fly_pilot.runway import ApproachConfig, Runway, approach_origin


def test_meters_per_degree_at_equator_and_mid_latitude() -> None:
    m_lat0, m_lon0 = meters_per_degree(0.0)
    m_lat37, m_lon37 = meters_per_degree(37.0)
    assert 110_000 < m_lat0 < 112_000
    assert 110_000 < m_lon0 < 112_000
    assert m_lon37 < m_lon0
    assert abs(m_lat37 - 110_950) < 500


def test_enu_roundtrip() -> None:
    frame = LocalFrame(37.0, -122.0, 0.0)
    east, north, up = 3200.0, -150.0, 250.0
    lat, lon, alt = frame.from_enu(east, north, up)
    e2, n2, u2 = frame.to_enu(lat, lon, alt)
    assert abs(e2 - east) < 0.05
    assert abs(n2 - north) < 0.05
    assert abs(u2 - up) < 1e-9


def test_runway_heading_east_puts_approach_west_of_threshold() -> None:
    runway = Runway(heading_deg=90.0)
    config = ApproachConfig(distance_m=3200.0, agl_m=250.0)
    lat, lon, alt = approach_origin(runway, config)
    east, north, up = runway.frame.to_enu(lat, lon, alt)
    along, right = runway_coordinates(east, north, runway.heading_deg)
    assert along < -3100.0
    assert abs(right) < 1.0
    assert abs(up - 250.0) < 0.01
    east_h, north_h = heading_components(90.0)
    assert abs(east_h - 1.0) < 1e-9
    assert abs(north_h) < 1e-9


def test_along_track_increases_when_flying_runway_heading() -> None:
    along0, _ = runway_coordinates(-3200.0, 0.0, 90.0)
    along1, right1 = runway_coordinates(-3100.0, 20.0, 90.0)
    assert along1 > along0
    # Heading 090: +north is left of track, so right-track is negative.
    assert abs(right1 + 20.0) < 1e-9
