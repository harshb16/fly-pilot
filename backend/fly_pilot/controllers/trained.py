"""Fixed MaleCNS + trained temporal decoder as an aircraft controller.

Path (nothing else):

    aircraft pose
      → canonical Python fly-view
        → FlyEyeEncoder
          → R1–R6 currents
            → full MaleCNS LIF
              → 100/260 ms DN rates
                → GRU decoder
                  → AircraftControls
                    → JSBSim

ExpertLandingController is not imported and must not appear in this module.
Aircraft telemetry is not a decoder input. This is not biological learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fly_pilot.brain.decoder import DecoderArtifact, default_checkpoint_path
from fly_pilot.brain.features import DECODER_INPUT_KIND
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.controllers.base import Controller
from fly_pilot.state import AircraftControls, AircraftObservation

# Small first-order low-pass on decoder outputs, applied in neural time
# (~50 Hz). Alpha 0.55 keeps the command responsive while damping single-step
# GRU chatter. This is an actuator convenience, not a runway-guidance law.
# Small first-order low-pass on decoder outputs, applied in neural time
# (~50 Hz). Alpha 0.70 keeps opening corrections (the expert's first 20 s)
# while damping single-step GRU chatter. Not a runway-guidance law.
SLEW_ALPHA = 0.70


class TrainedMaleCNSController(Controller):
    """FLY CONTROL authority. Decoder outputs go to JSBSim."""

    name = "fly_control"
    kind = "fixed_malecns_plus_trained_temporal_decoder"
    biological_learning = False
    uses_expert = False
    uses_telemetry_as_decoder_input = False

    def __init__(
        self,
        observer: ObservingMaleCNS,
        artifact: DecoderArtifact,
        *,
        slew_alpha: float = SLEW_ALPHA,
    ) -> None:
        if observer.dn.spec.n == 0:
            raise ValueError("observer has no descending neurons")
        artifact.assert_dn_ordering(observer.dn.spec.body_ids)
        if artifact.config.n_descending != observer.dn.spec.n:
            raise ValueError(
                f"decoder n_descending={artifact.config.n_descending} "
                f"!= observer {observer.dn.spec.n}"
            )
        if tuple(artifact.config.windows) != tuple(observer.dn.windows):
            raise ValueError("decoder DN windows do not match the observer extractor")
        self.observer = observer
        self.artifact = artifact
        self.decoder = artifact.model
        self.decoder.eval()
        self.slew_alpha = float(slew_alpha)
        self._obs: AircraftObservation | None = None
        self._decoded_step = -1
        self._raw = AircraftControls(throttle=0.4)
        self._filtered = AircraftControls(throttle=0.4)
        self.decoder.reset_state()

    @classmethod
    def load(
        cls,
        observer: ObservingMaleCNS,
        path: Path | None = None,
        *,
        slew_alpha: float = SLEW_ALPHA,
    ) -> "TrainedMaleCNSController":
        artifact = DecoderArtifact.load(path or default_checkpoint_path())
        return cls(observer, artifact, slew_alpha=slew_alpha)

    @classmethod
    def untrained(cls, observer: ObservingMaleCNS, seed: int = 0) -> "TrainedMaleCNSController":
        """Random-weight decoder for tests. Not a flight-capable policy."""
        from fly_pilot.brain.decoder import CausalTemporalDecoder, DecoderConfig

        rng = np.random.default_rng(seed)
        torch_seed = int(rng.integers(0, 2**31 - 1))
        import torch

        torch.manual_seed(torch_seed)
        config = DecoderConfig(
            n_descending=int(observer.dn.spec.n),
            windows=tuple(observer.dn.windows),
        )
        model = CausalTemporalDecoder(config)
        artifact = DecoderArtifact(
            config=config,
            model=model,
            dn_body_ids=np.asarray(observer.dn.spec.body_ids, dtype=np.int64),
            training_seed=seed,
            dataset={"synthetic": True},
        )
        return cls(observer, artifact)

    def reset(self) -> None:
        self.decoder.reset_state()
        self._decoded_step = -1
        self._raw = AircraftControls(throttle=0.4)
        self._filtered = AircraftControls(throttle=0.4)
        self._obs = None

    def observe(self, observation: AircraftObservation) -> None:
        self._obs = observation

    def _decoder_features(self) -> np.ndarray:
        vec = self.observer.dn.decoder_feature_vector()
        if vec.size != self.artifact.config.input_dim:
            raise ValueError(
                f"DN feature size {vec.size} != decoder input {self.artifact.config.input_dim}"
            )
        return vec

    def _slew(self, raw: AircraftControls) -> AircraftControls:
        a = self.slew_alpha
        prev = self._filtered
        return AircraftControls(
            aileron=a * raw.aileron + (1.0 - a) * prev.aileron,
            elevator=a * raw.elevator + (1.0 - a) * prev.elevator,
            rudder=a * raw.rudder + (1.0 - a) * prev.rudder,
            throttle=a * raw.throttle + (1.0 - a) * prev.throttle,
        ).clamped()

    def act(self) -> AircraftControls:
        step = self.observer.last_step
        neural_step = -1 if step is None else int(step.neural_step)
        if step is not None and neural_step != self._decoded_step:
            features = self._decoder_features()
            self._raw = self.decoder.controls_from_features(features)
            self._filtered = self._slew(self._raw)
            self._decoded_step = neural_step
        return self._filtered

    def telemetry(self) -> dict[str, Any]:
        hidden = self.decoder.hidden_summary()
        fly = self.observer.hud_payload()
        fly["controlling"] = True
        fly["label"] = "FLY CONTROL — FIXED MALECNS + TRAINED TEMPORAL DECODER"
        fly["mode"] = "FLY CONTROL"
        return {
            "kind": self.kind,
            "label": "FLY CONTROL — fixed MaleCNS + trained temporal decoder (not biological learning)",
            "male_cns": True,
            "biological_learning": False,
            "expert_in_loop": False,
            "decoder_input_kind": DECODER_INPUT_KIND,
            "decoder_uses_telemetry": False,
            "aileron": self._filtered.aileron,
            "elevator": self._filtered.elevator,
            "rudder": self._filtered.rudder,
            "throttle": self._filtered.throttle,
            "raw_aileron": self._raw.aileron,
            "raw_elevator": self._raw.elevator,
            "raw_rudder": self._raw.rudder,
            "raw_throttle": self._raw.throttle,
            "slew_alpha": self.slew_alpha,
            "decoded_neural_step": self._decoded_step,
            **hidden,
            "fly": fly,
        }
