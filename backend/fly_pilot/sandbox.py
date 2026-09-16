"""Landing sandbox: controller + JSBSim + optional MaleCNS.

Control authority is ManualController, ExpertLandingController, or
TrainedMaleCNSController (FLY CONTROL). When ``mode == "expert_observing"``,
MaleCNS watches the rendered approach but ExpertLandingController still flies.
When ``mode == "fly_control"``, decoder outputs go to JSBSim. The expert is
not in that path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fly_pilot.aircraft import Cessna172
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.brain.scheduler import SimScheduler
from fly_pilot.brain.vision.scene import FlyViewPose
from fly_pilot.controllers.base import Controller
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.manual import ManualController
from fly_pilot.controllers.trained import TrainedMaleCNSController
from fly_pilot.episode import EpisodeMonitor
from fly_pilot.initial_conditions import DECODER_SPAWN, SpawnRandomization, SpawnState, sample_spawn
from fly_pilot.runway import ApproachConfig, Runway
from fly_pilot.state import AircraftControls, AircraftObservation, EpisodeInfo

ALLOWED_CONTROLLERS = ("manual", "expert", "expert_observing", "fly_control")
CONTROL_AUTHORITY = ("manual", "expert", "fly_control")
UNIMPLEMENTED_MALE_CNS = ("malecns", "male_cns")


@dataclass
class SandboxSnapshot:
    observation: AircraftObservation
    episode: EpisodeInfo
    controller_name: str
    paused: bool
    spawn_seed: int | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    observing: bool = False
    fly_observing: dict[str, Any] = field(default_factory=dict)
    applied_controls: AircraftControls | None = None

    @property
    def ui_mode(self) -> str:
        return self.mode or self.controller_name


class LandingSandbox:
    def __init__(
        self,
        controller: Controller | None = None,
        runway: Runway | None = None,
        approach: ApproachConfig | None = None,
        randomize_spawns: bool | None = None,
        observer: ObservingMaleCNS | None = None,
        observing: bool = False,
        spawn_spec: SpawnRandomization | None = None,
        decoder_path: Path | None = None,
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
        self.observer = observer
        self.observing = observing and observer is not None
        self.scheduler = SimScheduler()
        self.last_applied_controls: AircraftControls | None = None
        self.decoder_path = decoder_path
        self.spawn_spec = spawn_spec
        if randomize_spawns is None:
            self.randomize_spawns = self.controller.name in ("expert", "fly_control")
        else:
            self.randomize_spawns = randomize_spawns
        self.reset()

    @property
    def mode(self) -> str:
        if isinstance(self.controller, TrainedMaleCNSController):
            return "fly_control"
        if self.observing:
            return "expert_observing"
        return self.controller.name

    def _neural_active(self) -> bool:
        return self.observing or isinstance(self.controller, TrainedMaleCNSController)

    def set_observer(self, observer: ObservingMaleCNS | None) -> None:
        self.observer = observer
        if observer is None:
            self.observing = False

    def _ensure_observer(self) -> ObservingMaleCNS:
        if self.observer is None:
            self.observer = ObservingMaleCNS.load(runway=self.runway)
        return self.observer

    def set_controller(self, name: str) -> SandboxSnapshot:
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        if key in ("expert_fly_observing", "expert+fly_observing"):
            key = "expert_observing"
        if key in ("trained", "trained_malecns", "trained_male_cns"):
            key = "fly_control"
        if key in UNIMPLEMENTED_MALE_CNS:
            raise ValueError(
                "Untrained MaleCNSController is not implemented. "
                "Choose 'manual', 'expert', 'expert_observing', or 'fly_control' "
                "(fixed MaleCNS + trained temporal decoder)."
            )
        if key not in ALLOWED_CONTROLLERS:
            raise ValueError(f"unknown controller {name!r}; expected {ALLOWED_CONTROLLERS}")
        if key == "manual":
            self.controller = ManualController(AircraftControls(throttle=self.approach.throttle))
            self.randomize_spawns = False
            self.observing = False
            self.spawn_spec = None
        elif key == "expert":
            self.controller = ExpertLandingController(self.runway)
            self.randomize_spawns = True
            self.observing = False
            self.spawn_spec = None
        elif key == "expert_observing":
            self.controller = ExpertLandingController(self.runway)
            self.randomize_spawns = True
            self._ensure_observer()
            self.observing = True
            self.spawn_spec = None
        else:
            observer = self._ensure_observer()
            self.controller = TrainedMaleCNSController.load(observer, self.decoder_path)
            self.randomize_spawns = True
            self.observing = False
            self.spawn_spec = DECODER_SPAWN
        if self.controller.name not in CONTROL_AUTHORITY:
            raise RuntimeError("control authority must remain manual, expert, or fly_control")
        if self.mode == "fly_control" and isinstance(self.controller, ExpertLandingController):
            raise RuntimeError("FLY CONTROL must not instantiate ExpertLandingController")
        return self.reset()

    def reset(self, seed: int | None = None) -> SandboxSnapshot:
        self.paused = False
        self.controller.reset()
        if isinstance(self.controller, ManualController):
            self.controller.set_pilot_input(AircraftControls(throttle=self.approach.throttle))
        self.episode.reset()
        self._carry_s = 0.0
        self.scheduler.reset()
        spawn: SpawnState | None = None
        spec = self.spawn_spec
        if seed is not None:
            self.spawn_seed = seed
            spawn = sample_spawn(self.runway, self.approach, seed=seed, spec=spec)
        elif self.randomize_spawns:
            self._expert_seed_counter += 1
            self.spawn_seed = self._expert_seed_counter
            spawn = sample_spawn(self.runway, self.approach, seed=self.spawn_seed, spec=spec)
        else:
            self.spawn_seed = None
        obs = self.aircraft.reset(spawn)
        self.controller.observe(obs)
        self.last_applied_controls = None
        if self._neural_active() and self.observer is not None:
            brain_seed = self.spawn_seed if self.spawn_seed is not None else self.observer.config.seed
            self.observer.reset(seed=brain_seed)
            self._run_neural_due(obs)
            self.controller.observe(obs)
            if isinstance(self.controller, TrainedMaleCNSController):
                self.last_applied_controls = self.controller.act()
        return self.snapshot(obs)

    def set_manual_controls(self, controls: AircraftControls) -> None:
        if not isinstance(self.controller, ManualController):
            return
        self.controller.set_pilot_input(controls)

    def _run_neural_due(self, obs: AircraftObservation) -> None:
        """Step MaleCNS for every scheduler tick due at this sim time."""
        if not self._neural_active() or self.observer is None:
            return
        due = self.scheduler.drain(obs.sim_time_s)
        if not due:
            return
        pose = FlyViewPose.from_observation(obs)
        atlas = None
        for events in due:
            if events.vision or atlas is None:
                from fly_pilot.brain.vision.scene import render_cubemap_atlas

                atlas = render_cubemap_atlas(pose, self.runway)
                self.observer.last_atlas = atlas
            if events.neural:
                self.observer.step_atlas(
                    atlas,
                    sim_time_s=obs.sim_time_s,
                    neural_step=events.neural_step,
                )

    def step_once(self) -> SandboxSnapshot:
        obs = self.aircraft.observe()
        if self._neural_active():
            # Vision → MaleCNS on the current pose, then the controller acts.
            # FLY CONTROL reads DN features from this tick; the expert is absent.
            self._run_neural_due(obs)
        self.controller.observe(obs)
        controls = self.controller.act()
        if self.mode == "fly_control" and isinstance(self.controller, ExpertLandingController):
            raise RuntimeError("FLY CONTROL leaked ExpertLandingController into act()")
        self.last_applied_controls = controls
        self.aircraft.apply_controls(controls)
        obs = self.aircraft.step(1)
        self.scheduler.note_physics(self.aircraft.dt)
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
                self.paused = True
                break
        return snapshot

    def snapshot(self, obs: AircraftObservation | None = None) -> SandboxSnapshot:
        observation = obs if obs is not None else self.aircraft.observe()
        debug: dict[str, Any] = {}
        telemetry = getattr(self.controller, "telemetry", None)
        if callable(telemetry):
            debug = dict(telemetry())
        fly: dict[str, Any] = {}
        if self.observing and self.observer is not None:
            fly = self.observer.hud_payload()
        elif isinstance(self.controller, TrainedMaleCNSController):
            fly = dict(debug.get("fly") or {})
        return SandboxSnapshot(
            observation=observation,
            episode=self.episode.info,
            controller_name=self.controller.name,
            paused=self.paused,
            spawn_seed=self.spawn_seed,
            debug=debug,
            mode=self.mode,
            observing=self.observing,
            fly_observing=fly,
            applied_controls=self.last_applied_controls,
        )
