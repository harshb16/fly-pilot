"""Landing sandbox: controller + JSBSim + episode monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fly_pilot.aircraft import Cessna172
from fly_pilot.controllers.base import Controller
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.manual import ManualController
from fly_pilot.episode import EpisodeMonitor
from fly_pilot.initial_conditions import SpawnState, sample_spawn
from fly_pilot.runway import ApproachConfig, Runway
from fly_pilot.state import AircraftControls, AircraftObservation, EpisodeInfo

ALLOWED_CONTROLLERS = ("manual", "expert")


@dataclass
class SandboxSnapshot:
    observation: AircraftObservation
    episode: EpisodeInfo
    controller_name: str
    paused: bool
    spawn_seed: int | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class LandingSandbox:
    def __init__(
        self,
        controller: Controller | None = None,
        runway: Runway | None = None,
        approach: ApproachConfig | None = None,
        randomize_spawns: bool | None = None,
    ) -> None:
        self.runway = runway or Runway()
        self.approach = approach or ApproachConfig()
        self.aircraft = Cessna172(self.runway, self.approach)
        self.controller: Controller = controller or ManualController(
            AircraftControls(throttle=self.approach.throttle)
        )
        self.episode = EpisodeMonitor(self.runway, self.approach)
        self.paused = False
        self._carry_s = 0.0
        self.spawn_seed: int | None = None
        self._expert_seed_counter = 0
        if randomize_spawns is None:
            self.randomize_spawns = isinstance(self.controller, ExpertLandingController)
        else:
            self.randomize_spawns = randomize_spawns
        self.reset()

    def set_controller(self, name: str) -> SandboxSnapshot:
        key = name.strip().lower()
        if key in ("malecns", "male_cns", "trained", "trained_malecns"):
            raise ValueError(
                "MaleCNS controllers are not implemented in this milestone. "
                "Choose 'manual' or 'expert' (conventional autopilot)."
            )
        if key not in ALLOWED_CONTROLLERS:
            raise ValueError(f"unknown controller {name!r}; expected {ALLOWED_CONTROLLERS}")
        if key == "manual":
            self.controller = ManualController(AircraftControls(throttle=self.approach.throttle))
            self.randomize_spawns = False
        else:
            self.controller = ExpertLandingController(self.runway)
            self.randomize_spawns = True
        return self.reset()

    def reset(self, seed: int | None = None) -> SandboxSnapshot:
        self.paused = False
        self.controller.reset()
        if isinstance(self.controller, ManualController):
            self.controller.set_pilot_input(AircraftControls(throttle=self.approach.throttle))
        self.episode.reset()
        self._carry_s = 0.0
        spawn: SpawnState | None = None
        if seed is not None:
            self.spawn_seed = seed
            spawn = sample_spawn(self.runway, self.approach, seed=seed)
        elif self.randomize_spawns:
            self._expert_seed_counter += 1
            self.spawn_seed = self._expert_seed_counter
            spawn = sample_spawn(self.runway, self.approach, seed=self.spawn_seed)
        else:
            self.spawn_seed = None
        obs = self.aircraft.reset(spawn)
        self.controller.observe(obs)
        return self.snapshot(obs)

    def set_manual_controls(self, controls: AircraftControls) -> None:
        if not isinstance(self.controller, ManualController):
            return
        self.controller.set_pilot_input(controls)

    def step_once(self) -> SandboxSnapshot:
        obs = self.aircraft.observe()
        self.controller.observe(obs)
        controls = self.controller.act()
        self.aircraft.apply_controls(controls)
        obs = self.aircraft.step(1)
        self.episode.update(obs)
        return self.snapshot(obs)

    def step_realtime(self, wall_dt: float, catchup_limit: int = 24) -> SandboxSnapshot:
        if self.paused:
            return self.snapshot()
        self._carry_s += max(0.0, wall_dt)
        n = int(self._carry_s / self.aircraft.dt)
        self._carry_s -= n * self.aircraft.dt
        n = min(n, catchup_limit)
        snapshot = self.snapshot()
        for _ in range(n):
            snapshot = self.step_once()
            if snapshot.episode.status.value != "in_progress":
                # Keep streaming the frozen outcome; do not keep integrating
                # a crashed/landed airframe into the ground.
                self.paused = True
                break
        return snapshot

    def snapshot(self, obs: AircraftObservation | None = None) -> SandboxSnapshot:
        observation = obs if obs is not None else self.aircraft.observe()
        debug: dict[str, Any] = {}
        telemetry = getattr(self.controller, "telemetry", None)
        if callable(telemetry):
            debug = dict(telemetry())
        return SandboxSnapshot(
            observation=observation,
            episode=self.episode.info,
            controller_name=self.controller.name,
            paused=self.paused,
            spawn_seed=self.spawn_seed,
            debug=debug,
        )
