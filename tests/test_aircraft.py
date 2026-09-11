"""JSBSim integration tests. These require the real c172p model."""

from __future__ import annotations

import math

import pytest

from fly_pilot.aircraft import Cessna172
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import AircraftControls, EpisodeStatus


@pytest.fixture(scope="module")
def aircraft() -> Cessna172:
    return Cessna172()


def test_reset_places_c172_on_short_final(aircraft: Cessna172) -> None:
    obs = aircraft.reset()
    assert 200.0 <= obs.alt_agl_m <= 400.0
    assert 2_000.0 <= -obs.along_m <= 4_000.0
    assert abs(obs.right_m) < 30.0
    assert 60.0 <= obs.airspeed_kts <= 85.0
    assert abs((obs.heading_deg - 90.0 + 180) % 360 - 180) < 8.0
    assert obs.on_ground is False


def test_elevator_nose_up_increases_pitch(aircraft: Cessna172) -> None:
    aircraft.reset()
    start = aircraft.observe().pitch_deg
    aircraft.apply_controls(AircraftControls(elevator=0.45, throttle=0.45))
    end = aircraft.step(120).pitch_deg  # 1 second at 120 Hz
    assert end > start + 3.0


def test_aileron_rolls_right(aircraft: Cessna172) -> None:
    aircraft.reset()
    start = aircraft.observe().roll_deg
    aircraft.apply_controls(AircraftControls(aileron=0.4, throttle=0.45))
    end = aircraft.step(120).roll_deg
    assert end > start + 4.0


def test_throttle_changes_are_visible_in_state(aircraft: Cessna172) -> None:
    aircraft.reset()
    aircraft.apply_controls(AircraftControls(throttle=0.8))
    obs = aircraft.step(10)
    assert obs.throttle == pytest.approx(0.8)
    assert aircraft.fdm["fcs/throttle-cmd-norm"] == pytest.approx(0.8)


def test_reset_restores_approach_state(aircraft: Cessna172) -> None:
    first = aircraft.reset()
    aircraft.apply_controls(AircraftControls(aileron=0.5, elevator=-0.3, throttle=1.0))
    aircraft.step(240)
    second = aircraft.reset()
    assert abs(second.alt_agl_m - first.alt_agl_m) < 5.0
    assert abs(second.along_m - first.along_m) < 20.0
    assert abs(second.roll_deg) < 2.0
    assert abs(second.airspeed_kts - first.airspeed_kts) < 5.0
    assert second.sim_time_s < 0.05


def test_sandbox_manual_loop_is_stable() -> None:
    sandbox = LandingSandbox()
    start = sandbox.snapshot().observation
    for _ in range(240):
        sandbox.set_manual_controls(AircraftControls(elevator=0.05, throttle=0.42))
        snap = sandbox.step_once()
        obs = snap.observation
        assert math.isfinite(obs.alt_agl_m)
        assert math.isfinite(obs.pitch_deg)
        assert math.isfinite(obs.airspeed_kts)
        if snap.episode.status is not EpisodeStatus.IN_PROGRESS:
            break
    end = sandbox.snapshot().observation
    # One second of small input should not NaN or teleport the airplane.
    assert abs(end.along_m - start.along_m) < 200.0
    assert 10.0 < end.alt_agl_m < 600.0


def test_telemetry_fields_are_populated(aircraft: Cessna172) -> None:
    obs = aircraft.reset()
    assert obs.lat_deg != 0.0
    assert obs.lon_deg != 0.0
    assert obs.extra["throttle_pos"] >= 0.0
    wow = obs.wow
    assert len(wow) == 3
