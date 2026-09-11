"""Landing sandbox: controller + JSBSim + episode monitor."""

from __future__ import annotations

from dataclasses import dataclass

from fly_pilot.aircraft import Cessna172
from fly_pilot.controllers.base import Controller
from fly_pilot.controllers.manual import ManualController
from fly_pilot.episode import EpisodeMonitor
from fly_pilot.runway import ApproachConfig, Runway
from fly_pilot.state import AircraftControls, AircraftObservation, EpisodeInfo


@dataclass
class SandboxSnapshot:
    observation: AircraftObservation
    episode: EpisodeInfo
    controller_name: str
    paused: bool


class LandingSandbox:
    def __init__(
        self,
        controller: Controller | None = None,
        runway: Runway | None = None,
        approach: ApproachConfig | None = None,
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
        self.reset()

    def reset(self) -> SandboxSnapshot:
        self.paused = False
        self.controller.reset()
        if isinstance(self.controller, ManualController):
            self.controller.set_pilot_input(AircraftControls(throttle=self.approach.throttle))
        self.episode.reset()
        self._carry_s = 0.0
        obs = self.aircraft.reset()
        self.controller.observe(obs)
        return self.snapshot(obs)

    def set_manual_controls(self, controls: AircraftControls) -> None:
        if not isinstance(self.controller, ManualController):
            raise TypeError("active controller is not ManualController")
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
        return SandboxSnapshot(
            observation=observation,
            episode=self.episode.info,
            controller_name=self.controller.name,
            paused=self.paused,
        )
