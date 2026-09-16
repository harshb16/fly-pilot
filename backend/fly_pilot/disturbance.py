"""Bounded inceptor pulses applied during some expert-observation episodes.

The ExpertLandingController still produces the supervised target. Pulses are
added only to the command that JSBSim actually receives, so the next expert
action is a recovery. No wind.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from fly_pilot.state import AircraftControls


@dataclass(frozen=True, slots=True)
class DisturbanceConfig:
    """Occasional small pulses. Magnitudes stay inside expert recovery."""

    episode_probability: float = 0.45
    max_pulses: int = 3
    min_alt_agl_m: float = 40.0
    min_start_s: float = 2.0
    duration_s: tuple[float, float] = (0.35, 1.0)
    gap_s: tuple[float, float] = (2.5, 6.0)
    aileron: tuple[float, float] = (-0.14, 0.14)
    elevator: tuple[float, float] = (-0.09, 0.09)
    rudder: tuple[float, float] = (-0.08, 0.08)
    throttle: tuple[float, float] = (-0.06, 0.06)


@dataclass(frozen=True, slots=True)
class DisturbancePulse:
    start_s: float
    end_s: float
    aileron: float
    elevator: float
    rudder: float
    throttle: float

    def active(self, sim_time_s: float) -> bool:
        return self.start_s <= sim_time_s < self.end_s

    def as_dict(self) -> dict[str, float]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "aileron": self.aileron,
            "elevator": self.elevator,
            "rudder": self.rudder,
            "throttle": self.throttle,
        }


class ControlDisturbance:
    """Deterministic pulse schedule derived from an episode seed."""

    def __init__(self, pulses: tuple[DisturbancePulse, ...] = ()) -> None:
        self.pulses = pulses

    @classmethod
    def plan(
        cls,
        seed: int,
        *,
        config: DisturbanceConfig | None = None,
        enabled: bool | None = None,
    ) -> "ControlDisturbance":
        cfg = config or DisturbanceConfig()
        rng = random.Random(seed ^ 0xA5A5_173)
        if enabled is None:
            enabled = rng.random() < cfg.episode_probability
        if not enabled:
            return cls(())
        pulses: list[DisturbancePulse] = []
        t = cfg.min_start_s + rng.uniform(0.0, 1.5)
        n_pulses = rng.randint(1, cfg.max_pulses)
        for _ in range(n_pulses):
            dur = rng.uniform(*cfg.duration_s)
            pulses.append(
                DisturbancePulse(
                    start_s=t,
                    end_s=t + dur,
                    aileron=rng.uniform(*cfg.aileron),
                    elevator=rng.uniform(*cfg.elevator),
                    rudder=rng.uniform(*cfg.rudder),
                    throttle=rng.uniform(*cfg.throttle),
                )
            )
            t += dur + rng.uniform(*cfg.gap_s)
        return cls(tuple(pulses))

    def offset(self, sim_time_s: float, alt_agl_m: float, min_alt_agl_m: float = 40.0) -> AircraftControls:
        if alt_agl_m < min_alt_agl_m:
            return AircraftControls(throttle=0.0)
        aileron = elevator = rudder = throttle = 0.0
        for pulse in self.pulses:
            if pulse.active(sim_time_s):
                aileron += pulse.aileron
                elevator += pulse.elevator
                rudder += pulse.rudder
                throttle += pulse.throttle
        return AircraftControls(aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle)

    def apply(
        self,
        expert: AircraftControls,
        *,
        sim_time_s: float,
        alt_agl_m: float,
        min_alt_agl_m: float = 40.0,
    ) -> AircraftControls:
        delta = self.offset(sim_time_s, alt_agl_m, min_alt_agl_m=min_alt_agl_m)
        return AircraftControls(
            aileron=expert.aileron + delta.aileron,
            elevator=expert.elevator + delta.elevator,
            rudder=expert.rudder + delta.rudder,
            throttle=expert.throttle + delta.throttle,
        ).clamped()

    def as_list(self) -> list[dict[str, float]]:
        return [p.as_dict() for p in self.pulses]
