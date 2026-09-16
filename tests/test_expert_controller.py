from fly_pilot.controllers.expert import ExpertLandingController, LandingPhase
from fly_pilot.controllers.manual import ManualController
from fly_pilot.episode import is_successful_touchdown
from fly_pilot.record_observing import synthetic_observer
from fly_pilot.runway import Runway
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import AircraftControls, AircraftObservation, EpisodeStatus

import pytest


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
        p_deg_s=0.0,
        q_deg_s=0.0,
        r_deg_s=0.0,
        beta_deg=0.0,
    )
    defaults.update(kwargs)
    return AircraftObservation(**defaults)


def test_expert_outputs_stay_in_bounds() -> None:
    ctl = ExpertLandingController()
    assert ctl.name == "expert"
    assert ctl.kind == "conventional_autopilot"
    ctl.reset()
    cases = [
        _obs(),
        _obs(right_m=80, roll_deg=-25, heading_deg=70, alt_agl_m=180, along_m=-2500),
        _obs(right_m=-90, roll_deg=30, heading_deg=110, airspeed_kts=55),
        _obs(alt_agl_m=12, along_m=40, vertical_speed_fpm=-600, pitch_deg=4),
        _obs(alt_agl_m=2, along_m=200, on_ground=True, wow=(True, True, False)),
        _obs(airspeed_kts=40, pitch_deg=15, roll_deg=-40, vertical_speed_fpm=-1200),
    ]
    for obs in cases:
        ctl.observe(obs)
        out = ctl.act()
        assert -1.0 <= out.aileron <= 1.0
        assert -1.0 <= out.elevator <= 1.0
        assert -1.0 <= out.rudder <= 1.0
        assert 0.0 <= out.throttle <= 1.0


def test_expert_telemetry_is_labelled_conventional() -> None:
    ctl = ExpertLandingController()
    ctl.observe(_obs(alt_agl_m=12, along_m=10))
    ctl.act()
    tel = ctl.telemetry()
    assert tel["kind"] == "conventional_autopilot"
    assert tel["male_cns"] is False
    assert "not biological" in str(tel["label"]).lower()
    assert tel["phase"] in {p.value for p in LandingPhase}
    assert "target_airspeed_kts" in tel
    assert "glideslope_error_deg" in tel
    assert "centerline_error_m" in tel


def test_expert_phase_flare_near_runway() -> None:
    ctl = ExpertLandingController()
    ctl.observe(_obs(sim_time_s=80, alt_agl_m=12, along_m=-20, airspeed_kts=65))
    ctl.act()
    assert ctl.phase is LandingPhase.FLARE


def test_sandbox_controller_mode_switch() -> None:
    sandbox = LandingSandbox()
    assert sandbox.controller.name == "manual"
    snap = sandbox.set_controller("expert")
    assert isinstance(sandbox.controller, ExpertLandingController)
    assert snap.controller_name == "expert"
    sandbox.set_controller("manual")
    assert isinstance(sandbox.controller, ManualController)


def test_sandbox_rejects_malecns_placeholder() -> None:
    sandbox = LandingSandbox()
    with pytest.raises(ValueError) as exc:
        sandbox.set_controller("malecns")
    sandbox.set_observer(synthetic_observer())
    snap = sandbox.set_controller("expert_observing")
    assert snap.observing is True
    assert sandbox.controller.name == "expert"


def test_successful_touchdown_classification() -> None:
    good = _obs(
        along_m=200,
        right_m=2.0,
        alt_agl_m=0.8,
        on_ground=True,
        wow=(True, True, False),
        vertical_speed_fpm=-250.0,
        roll_deg=2.0,
        pitch_deg=3.0,
    )
    assert is_successful_touchdown(good, Runway()) is True
    hard = _obs(
        along_m=200,
        right_m=0.0,
        alt_agl_m=0.8,
        on_ground=True,
        vertical_speed_fpm=-900.0,
        roll_deg=1.0,
        pitch_deg=2.0,
    )
    assert is_successful_touchdown(hard, Runway()) is False
    off = _obs(
        along_m=200,
        right_m=40.0,
        alt_agl_m=0.8,
        on_ground=True,
        vertical_speed_fpm=-200.0,
        roll_deg=0.0,
        pitch_deg=2.0,
    )
    assert is_successful_touchdown(off, Runway()) is False
