"""Seeded, modest randomization of the short-final spawn.

No wind. Ranges stay near the Milestone 1 approach so the expert
controller is a solvability baseline, not a storm-penetration demo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from fly_pilot.guidance import spawn_enu
from fly_pilot.runway import ApproachConfig, Runway


@dataclass(frozen=True, slots=True)
class SpawnState:
    """Initial-condition offsets applied at JSBSim `run_ic()`."""

    distance_m: float
    right_m: float
    agl_m: float
    airspeed_kts: float
    heading_deg: float
    gamma_deg: float
    alpha_deg: float
    roll_deg: float
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class SpawnRandomization:
    """Half-range (or absolute range) around the nominal approach."""

    distance_m: tuple[float, float] = (-150.0, 150.0)
    right_m: tuple[float, float] = (-80.0, 80.0)
    agl_m: tuple[float, float] = (-30.0, 30.0)
    airspeed_kts: tuple[float, float] = (-5.0, 5.0)
    heading_error_deg: tuple[float, float] = (-8.0, 8.0)
    gamma_deg: tuple[float, float] = (-1.0, 1.0)
    alpha_deg: tuple[float, float] = (-1.5, 1.5)
    roll_deg: tuple[float, float] = (-8.0, 8.0)


def nominal_spawn(runway: Runway, approach: ApproachConfig) -> SpawnState:
    return SpawnState(
        distance_m=approach.distance_m,
        right_m=0.0,
        agl_m=approach.agl_m,
        airspeed_kts=approach.airspeed_kts,
        heading_deg=runway.heading_deg,
        gamma_deg=approach.flight_path_deg,
        alpha_deg=approach.alpha_deg,
        roll_deg=0.0,
        seed=None,
    )


# Wider than the Milestone 2 solvability envelope, still inside states the
# conventional expert can recover from. Used for decoder-training collection
# and held-out autonomous fly-control evaluation. No wind.
DECODER_SPAWN = SpawnRandomization(
    distance_m=(-250.0, 250.0),
    right_m=(-140.0, 140.0),
    agl_m=(-50.0, 50.0),
    airspeed_kts=(-8.0, 8.0),
    heading_error_deg=(-12.0, 12.0),
    gamma_deg=(-1.5, 1.5),
    alpha_deg=(-2.0, 2.0),
    roll_deg=(-14.0, 14.0),
)


def sample_spawn(
    runway: Runway,
    approach: ApproachConfig | None = None,
    rng: random.Random | None = None,
    spec: SpawnRandomization | None = None,
    seed: int | None = None,
) -> SpawnState:
    """Draw one modest final-approach spawn. The same seed is bit-for-bit repeatable."""
    approach = approach or ApproachConfig()
    spec = spec or SpawnRandomization()
    if rng is None:
        rng = random.Random(seed)
    elif seed is not None:
        rng.seed(seed)

    def _u(lo_hi: tuple[float, float]) -> float:
        return rng.uniform(lo_hi[0], lo_hi[1])

    return SpawnState(
        distance_m=approach.distance_m + _u(spec.distance_m),
        right_m=_u(spec.right_m),
        agl_m=approach.agl_m + _u(spec.agl_m),
        airspeed_kts=approach.airspeed_kts + _u(spec.airspeed_kts),
        heading_deg=runway.heading_deg + _u(spec.heading_error_deg),
        gamma_deg=approach.flight_path_deg + _u(spec.gamma_deg),
        alpha_deg=approach.alpha_deg + _u(spec.alpha_deg),
        roll_deg=_u(spec.roll_deg),
        seed=seed,
    )


def spawn_geodetic(runway: Runway, spawn: SpawnState) -> tuple[float, float, float]:
    """Geodetic (lat, lon, alt_m) for a SpawnState."""
    east, north, up = spawn_enu(runway, spawn.distance_m, spawn.right_m, spawn.agl_m)
    return runway.frame.from_enu(east, north, up)
