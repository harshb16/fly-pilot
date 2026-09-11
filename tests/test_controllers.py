from fly_pilot.controllers.manual import ManualController
from fly_pilot.state import AircraftControls, AircraftObservation


def _obs(**kwargs) -> AircraftObservation:
    defaults = dict(
        sim_time_s=1.0,
        lat_deg=37.0,
        lon_deg=-122.0,
        alt_msl_m=200.0,
        alt_agl_m=200.0,
        east_m=-2000.0,
        north_m=0.0,
        up_m=200.0,
        along_m=-2000.0,
        right_m=0.0,
        airspeed_kts=70.0,
        groundspeed_kts=68.0,
        vertical_speed_fpm=-400.0,
        pitch_deg=-2.0,
        roll_deg=0.0,
        heading_deg=90.0,
        alpha_deg=4.0,
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.4,
        on_ground=False,
        wow=(False, False, False),
    )
    defaults.update(kwargs)
    return AircraftObservation(**defaults)


def test_manual_controller_clamps_and_returns_pilot_input() -> None:
    ctl = ManualController()
    assert ctl.name == "manual"
    ctl.set_pilot_input(AircraftControls(aileron=2, elevator=-3, rudder=0.25, throttle=1.5))
    out = ctl.act()
    assert out.aileron == 1.0
    assert out.elevator == -1.0
    assert out.rudder == 0.25
    assert out.throttle == 1.0
    ctl.reset()
    assert ctl.act().throttle == 0.4


def test_manual_observe_does_not_invent_controls() -> None:
    ctl = ManualController(AircraftControls(throttle=0.3))
    ctl.observe(_obs())
    assert ctl.act().throttle == 0.3
    assert ctl.act().aileron == 0.0
