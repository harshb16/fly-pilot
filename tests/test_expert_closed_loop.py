"""Closed-loop ExpertLandingController against live JSBSim c172p."""

from __future__ import annotations

import pytest

from fly_pilot.aircraft import Cessna172
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.evaluate import run_episode
from fly_pilot.initial_conditions import sample_spawn
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import AircraftControls, EpisodeStatus


@pytest.fixture(scope="module")
def sandbox() -> LandingSandbox:
    return LandingSandbox(controller=ExpertLandingController(), randomize_spawns=True)


def test_engine_produces_thrust_after_reset() -> None:
    ac = Cessna172()
    obs = ac.reset()
    rpm = obs.extra["engine_rpm"]
    assert rpm > 800.0
    ac.apply_controls(AircraftControls(throttle=0.7, elevator=0.1))
    later = ac.step(120)
    assert later.extra["engine_rpm"] > 1000.0
    assert later.extra["thrust_lbs"] > 20.0


def test_seeded_jsbsim_reset_is_repeatable() -> None:
    ac = Cessna172()
    spawn = sample_spawn(ac.runway, seed=42)
    a = ac.reset(spawn)
    b = ac.reset(spawn)
    assert abs(a.right_m - b.right_m) < 2.0
    assert abs(a.alt_agl_m - b.alt_agl_m) < 3.0
    assert abs(a.airspeed_kts - b.airspeed_kts) < 3.0
    assert abs(((a.heading_deg - b.heading_deg + 180) % 360) - 180) < 3.0


def test_seeded_spawn_applies_lateral_offset() -> None:
    ac = Cessna172()
    spawn = sample_spawn(ac.runway, seed=3)
    obs = ac.reset(spawn)
    assert abs(obs.right_m - spawn.right_m) < 8.0
    assert abs(obs.alt_agl_m - spawn.agl_m) < 8.0


def test_expert_lands_nominal_approach(sandbox: LandingSandbox) -> None:
    metrics = run_episode(sandbox, seed=0, episode_id=0)
    assert metrics.outcome == EpisodeStatus.LANDED.value
    assert abs(metrics.centerline_error_m) < 8.0
    assert 0.0 <= metrics.along_m <= 1200.0
    assert -metrics.touchdown_vertical_speed_fpm <= 700.0
    assert abs(metrics.roll_deg) <= 12.0


def test_expert_closed_loop_majority_success(sandbox: LandingSandbox) -> None:
    outcomes = [run_episode(sandbox, seed=10 + i, episode_id=i) for i in range(5)]
    wins = sum(1 for m in outcomes if m.success)
    assert wins >= 4, [m.outcome for m in outcomes]
