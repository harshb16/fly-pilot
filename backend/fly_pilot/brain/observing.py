"""MaleCNS as a visual observer of expert landings.

EXPERT + FLY OBSERVING: ExpertLandingController flies the Cessna. This module
encodes the rendered fly-view into R1–R6 currents, steps MaleCNS, and records
features. It has **no** ``act()`` and must never produce ``AircraftControls``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fly_pilot.brain.config import LIFConfig, default_data_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.features import (
    DescendingNeuronFeatureExtractor,
    VisualPopulationTracker,
    descending_population,
    visual_debug_populations,
)
from fly_pilot.brain.model import MaleCNSLIF, StepResult
from fly_pilot.brain.scheduler import Schedule, SimScheduler
from fly_pilot.brain.stimulation import CurrentStimulation
from fly_pilot.brain.vision.encoder import EncoderConfig, FlyEyeEncoder
from fly_pilot.brain.vision.geometry import CameraCalibration
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.brain.vision.scene import FlyViewPose, render_cubemap_atlas
from fly_pilot.brain.vision.stimulus import RetinalStimulusFrame
from fly_pilot.runway import Runway
from fly_pilot.state import AircraftObservation

# Hard isolation: this observer is not a Controller and does not fly the plane.
CONTROLS_AIRCRAFT = False
MODE_LABEL = "EXPERT + FLY OBSERVING"
NOT_CONTROLLING_LABEL = "FLY OBSERVING — NOT CONTROLLING"


@dataclass
class ObservingStep:
    neural_step: int
    sim_time_s: float
    retinal: RetinalStimulusFrame
    n_spikes: int
    spike_checksum: str
    n_outgoing_edges: int
    dn_features: dict[str, Any]
    visual_rates: dict[str, float]
    encoder_summary: dict[str, float | int | str]
    spikes_per_sec_global: float


class ObservingMaleCNS:
    """Vision → MaleCNS. Explicitly not an aircraft controller."""

    name = "observing_malecns"
    controls_aircraft = CONTROLS_AIRCRAFT

    def __init__(
        self,
        connectome: Connectome,
        *,
        config: LIFConfig | None = None,
        mapping: PhotoreceptorMap | None = None,
        encoder_config: EncoderConfig | None = None,
        schedule: Schedule | None = None,
        runway: Runway | None = None,
    ) -> None:
        self.connectome = connectome
        self.config = config or LIFConfig()
        self.net = MaleCNSLIF(connectome, self.config)
        self.mapping = mapping or PhotoreceptorMap.from_connectome(connectome)
        self.encoder = FlyEyeEncoder(self.mapping, encoder_config)
        self.calibration = CameraCalibration()
        self.dn = DescendingNeuronFeatureExtractor(
            descending_population(connectome),
            dt=self.config.dt,
        )
        self.visual = VisualPopulationTracker(
            visual_debug_populations(connectome),
            dt=self.config.dt,
        )
        self.scheduler = SimScheduler(schedule or Schedule())
        self.runway = runway or Runway()
        self.last_step: ObservingStep | None = None
        self.last_atlas: np.ndarray | None = None

    @classmethod
    def load(cls, data_dir: Path | None = None, **kwargs) -> "ObservingMaleCNS":
        return cls(Connectome.load(data_dir or default_data_dir()), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        self.net.reset(seed=seed if seed is not None else self.config.seed)
        self.encoder.reset()
        self.dn.reset()
        self.visual.reset()
        self.scheduler.reset()
        self.last_step = None
        self.last_atlas = None

    def step_atlas(
        self,
        atlas: np.ndarray,
        *,
        sim_time_s: float,
        neural_step: int | None = None,
    ) -> ObservingStep:
        step = self.net.step_count if neural_step is None else int(neural_step)
        retinal = self.encoder.encode(atlas, neural_step=step, sim_time_s=sim_time_s)
        stim = CurrentStimulation(
            indices=self.mapping.indices,
            amplitudes=retinal.currents,
            label="r1r6_retina",
        )
        result = self.net.step(stim)
        return self._pack(result, retinal, sim_time_s)

    def step_pose(
        self,
        pose: FlyViewPose,
        *,
        sim_time_s: float,
        neural_step: int | None = None,
        runway: Runway | None = None,
    ) -> ObservingStep:
        atlas = render_cubemap_atlas(pose, runway or self.runway)
        self.last_atlas = atlas
        return self.step_atlas(atlas, sim_time_s=sim_time_s, neural_step=neural_step)

    def step_observation(self, obs: AircraftObservation, *, extra_yaw_deg: float = 0.0) -> ObservingStep:
        pose = FlyViewPose.from_observation(obs, extra_yaw_deg=extra_yaw_deg)
        return self.step_pose(pose, sim_time_s=obs.sim_time_s)

    def _pack(self, result: StepResult, retinal: RetinalStimulusFrame, sim_time_s: float) -> ObservingStep:
        dn_features = self.dn.update(result.fired)
        visual_rates = self.visual.update(result.fired)
        global_hz = float(result.n_spikes) / (self.connectome.n_neurons * self.config.dt) if self.connectome.n_neurons else 0.0
        packed = ObservingStep(
            neural_step=self.net.step_count - 1,
            sim_time_s=sim_time_s,
            retinal=retinal,
            n_spikes=result.n_spikes,
            spike_checksum=result.checksum,
            n_outgoing_edges=result.n_outgoing_edges,
            dn_features=dn_features,
            visual_rates=visual_rates,
            encoder_summary=self.encoder.summary(retinal),
            spikes_per_sec_global=global_hz * self.connectome.n_neurons,
        )
        self.last_step = packed
        return packed

    def hud_payload(self) -> dict[str, Any]:
        """Compact telemetry for the browser. Never includes inceptors."""
        step = self.last_step
        dn = self.dn.compact_summary()
        vis = {k: round(v, 4) for k, v in (step.visual_rates if step else {}).items() if k.endswith("_hz")}
        enc = step.encoder_summary if step else {}
        return {
            "controlling": False,
            "label": NOT_CONTROLLING_LABEL,
            "mode": MODE_LABEL,
            "neural_step": None if step is None else step.neural_step,
            "sim_time_s": None if step is None else step.sim_time_s,
            "n_spikes": 0 if step is None else step.n_spikes,
            "spikes_per_sec": 0.0 if step is None else step.spikes_per_sec_global,
            "n_outgoing_edges": 0 if step is None else step.n_outgoing_edges,
            "spike_checksum": "" if step is None else step.spike_checksum,
            "n_r1r6": self.mapping.n_receptors,
            "n_descending": self.dn.spec.n,
            "retina": {
                "mean_luminance": enc.get("mean_luminance", 0.0),
                "mean_current": enc.get("mean_current", 0.0),
                "mean_temporal": enc.get("mean_temporal", 0.0),
                "left_mean_luminance": enc.get("left_mean_luminance", 0.0),
                "right_mean_luminance": enc.get("right_mean_luminance", 0.0),
                "current_sha256": enc.get("current_sha256", ""),
            },
            "descending": dn,
            "visual_rates_hz": vis,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "controls_aircraft": self.controls_aircraft,
            "mode": MODE_LABEL,
            "lif": {
                "dt": self.config.dt,
                "tau_m": self.config.tau_m,
                "tonic_current": self.config.tonic_current,
                "synaptic_gain": self.config.synaptic_gain,
                "background_rate_hz": self.config.background_rate_hz,
                "background_amplitude": self.config.background_amplitude,
                "seed": self.config.seed,
                "normalize_incoming": self.config.normalize_incoming,
            },
            "schedule": self.scheduler.schedule.as_dict(),
            "camera": self.calibration.as_dict(),
            "encoder": self.encoder.config.as_dict(),
            "photoreceptors": self.mapping.as_dict(),
            "descending": self.dn.spec.as_dict(),
            "visual_populations": {name: spec.as_dict() for name, spec in self.visual.populations.items()},
            "n_neurons": self.connectome.n_neurons,
            "n_edges": self.connectome.n_edges,
            "integrity": (
                "ObservingMaleCNS encodes a rendered fly-view into MaleCNS LIF "
                "dynamics. It has no act() and is not an aircraft controller. "
                "FLY CONTROL, when selected, reads DN features via an external decoder."
            ),
        }
