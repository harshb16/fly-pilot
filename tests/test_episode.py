from fly_pilot.episode import EpisodeMonitor
from fly_pilot.state import AircraftObservation, EpisodeStatus


def _obs(**kwargs) -> AircraftObservation:
    defaults = dict(
        sim_time_s=10.0,
        lat_deg=37.0,
        lon_deg=-122.0,
        alt_msl_m=5.0,
        alt_agl_m=5.0,
        east_m=100.0,
        north_m=0.0,
        up_m=5.0,
        along_m=100.0,
        right_m=0.0,
        airspeed_kts=60.0,
        groundspeed_kts=58.0,
        vertical_speed_fpm=-200.0,
        pitch_deg=2.0,
        roll_deg=1.0,
        heading_deg=90.0,
        alpha_deg=5.0,
        aileron=0.0,
        elevator=0.2,
        rudder=0.0,
        throttle=0.2,
        on_ground=False,
        wow=(False, False, False),
    )
    defaults.update(kwargs)
    return AircraftObservation(**defaults)


def test_in_progress_on_short_final() -> None:
    monitor = EpisodeMonitor()
    info = monitor.update(_obs(along_m=-400, alt_agl_m=40, on_ground=False))
    assert info.status is EpisodeStatus.IN_PROGRESS


def test_successful_touchdown_on_runway() -> None:
    monitor = EpisodeMonitor()
    monitor.update(_obs(along_m=200, alt_agl_m=8, on_ground=False))
    info = monitor.update(
        _obs(
            along_m=210,
            alt_agl_m=1.0,
            on_ground=True,
            wow=(True, True, False),
            vertical_speed_fpm=-250.0,
            roll_deg=2.0,
            pitch_deg=3.0,
        )
    )
    assert info.status is EpisodeStatus.LANDED
    assert info.touchdown_fpm == -250.0


def test_hard_landing_is_a_crash() -> None:
    monitor = EpisodeMonitor()
    monitor.update(_obs(along_m=200, on_ground=False))
    info = monitor.update(
        _obs(
            along_m=210,
            alt_agl_m=0.5,
            on_ground=True,
            wow=(True, True, True),
            vertical_speed_fpm=-1400.0,
        )
    )
    assert info.status is EpisodeStatus.CRASHED
    assert "hard landing" in (info.reason or "")


def test_off_runway_ground_contact_is_a_crash() -> None:
    monitor = EpisodeMonitor()
    monitor.update(_obs(along_m=100, right_m=0, on_ground=False))
    info = monitor.update(
        _obs(
            along_m=100,
            right_m=80,
            alt_agl_m=0.4,
            on_ground=True,
            wow=(True, False, False),
            vertical_speed_fpm=-200.0,
        )
    )
    assert info.status is EpisodeStatus.CRASHED


def test_past_runway_is_failed_approach() -> None:
    monitor = EpisodeMonitor()
    info = monitor.update(_obs(along_m=1600, alt_agl_m=80, on_ground=False))
    assert info.status is EpisodeStatus.FAILED_APPROACH


def test_lateral_deviation_is_out_of_bounds() -> None:
    monitor = EpisodeMonitor()
    info = monitor.update(_obs(along_m=-1000, right_m=1200, alt_agl_m=200, on_ground=False))
    assert info.status is EpisodeStatus.OUT_OF_BOUNDS


def test_terminal_state_does_not_change() -> None:
    monitor = EpisodeMonitor()
    monitor.update(_obs(along_m=200, on_ground=False))
    first = monitor.update(
        _obs(along_m=210, on_ground=True, wow=(True, True, False), vertical_speed_fpm=-200)
    )
    second = monitor.update(_obs(along_m=400, on_ground=True, vertical_speed_fpm=-2000))
    assert first.status is EpisodeStatus.LANDED
    assert second.status is EpisodeStatus.LANDED
