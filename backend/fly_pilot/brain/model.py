"""Simplified LIF dynamics on a MaleCNS-derived sparse graph.

Connectivity is measured MaleCNS anatomy. The voltage update is a modeling
choice inspired by current MaleCNS demos (Fly64), not a validated biophysical
model of a living fruit fly.

Each step:

    previous spikes
    → weighted recurrent input  (W @ spikes, W from synapse counts × sign)
    → external stimulation
    → tonic drive + seeded Bernoulli background
    → leaky membrane update
    → threshold, spike, reset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from fly_pilot.brain.config import LIFConfig
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.replay import spike_checksum
from fly_pilot.brain.stimulation import CurrentStimulation, PulseStimulation
from fly_pilot.brain.telemetry import StepTelemetry, population_spike_count, rate_hz


@dataclass
class StepResult:
    fired: np.ndarray
    n_spikes: int
    n_background_events: int
    n_stimulated: int
    checksum: str
    n_outgoing_edges: int = 0


class MaleCNSLIF:
    """Point-neuron LIF over the full prepared connectome.

    Recurrent weights are stored as float32 CSR. The matvec is vectorized;
    there is no Python loop over edges. Setting ``recurrent_enabled=False``
    zeros the anatomical input for ablation tests without deleting edges.
    """

    def __init__(
        self,
        connectome: Connectome,
        config: LIFConfig | None = None,
        *,
        recurrent_enabled: bool = True,
    ) -> None:
        self.connectome = connectome
        self.config = config or LIFConfig()
        self.recurrent_enabled = recurrent_enabled
        self.n = connectome.n_neurons
        self.weights = connectome.signed_normalized_weights(self.config)
        if not sparse.isspmatrix_csr(self.weights):
            self.weights = self.weights.tocsr()
        # CSC lets each step walk only the outgoing edges of cells that spiked.
        # That is sparse evaluation of the full graph, not pruning.
        self.weights_csc = self.weights.tocsc()
        self.voltage = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.step_count = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.last_recurrent = np.zeros(self.n, dtype=np.float32)
        self.last_background = np.zeros(self.n, dtype=np.uint8)
        self.last_external_count = 0
        self.last_outgoing_edges = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config = LIFConfig(
                dt=self.config.dt,
                tau_m=self.config.tau_m,
                v_threshold=self.config.v_threshold,
                v_reset=self.config.v_reset,
                tonic_current=self.config.tonic_current,
                synaptic_gain=self.config.synaptic_gain,
                background_rate_hz=self.config.background_rate_hz,
                background_amplitude=self.config.background_amplitude,
                normalize_incoming=self.config.normalize_incoming,
                seed=seed,
            )
        self.voltage[:] = 0
        self.spikes[:] = 0
        self.step_count = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.last_recurrent[:] = 0
        self.last_background[:] = 0
        self.last_external_count = 0
        self.last_outgoing_edges = 0

    def step(self, stimulation: PulseStimulation | CurrentStimulation | None = None) -> StepResult:
        cfg = self.config
        n_outgoing = 0
        if self.recurrent_enabled:
            spiking = np.flatnonzero(self.spikes)
            if spiking.size:
                starts = self.weights_csc.indptr[spiking]
                ends = self.weights_csc.indptr[spiking + 1]
                n_outgoing = int((ends - starts).sum())
                recurrent = np.asarray(
                    self.weights_csc[:, spiking].sum(axis=1),
                    dtype=np.float32,
                ).ravel()
                recurrent *= np.float32(cfg.synaptic_gain)
            else:
                recurrent = np.zeros(self.n, dtype=np.float32)
        else:
            recurrent = np.zeros(self.n, dtype=np.float32)
        self.last_recurrent = recurrent
        self.last_outgoing_edges = n_outgoing

        self.voltage *= np.float32(cfg.decay)
        self.voltage += recurrent
        if cfg.tonic_current:
            self.voltage += np.float32(cfg.tonic_current)

        n_background = 0
        if cfg.background_rate_hz > 0 and cfg.background_amplitude != 0:
            events = self.rng.random(self.n) < (cfg.background_rate_hz * cfg.dt)
            self.last_background = events.astype(np.uint8, copy=False)
            n_background = int(events.sum())
            if n_background:
                self.voltage[events] += np.float32(cfg.background_amplitude)
        else:
            self.last_background = np.zeros(self.n, dtype=np.uint8)

        n_stimulated = 0
        if stimulation is not None:
            n_stimulated = stimulation.apply(self.voltage, self.step_count)
        self.last_external_count = n_stimulated

        fired = self.voltage >= np.float32(cfg.v_threshold)
        self.voltage[fired] = np.float32(cfg.v_reset)
        self.spikes = fired.astype(np.float32, copy=False)
        result = StepResult(
            fired=fired,
            n_spikes=int(fired.sum()),
            n_background_events=n_background,
            n_stimulated=n_stimulated,
            checksum=spike_checksum(fired),
            n_outgoing_edges=n_outgoing,
        )
        self.step_count += 1
        return result

    def telemetry(
        self,
        result: StepResult,
        populations: dict[str, np.ndarray] | None = None,
        stimulated: np.ndarray | None = None,
    ) -> StepTelemetry:
        cfg = self.config
        pop_spikes: dict[str, int] = {}
        pop_rates: dict[str, float] = {}
        if populations:
            for name, indices in populations.items():
                count = population_spike_count(result.fired, indices)
                pop_spikes[name] = count
                pop_rates[name] = rate_hz(count, int(indices.size), cfg.dt)
        stimulated_spikes = 0
        if stimulated is not None and stimulated.size:
            stimulated_spikes = population_spike_count(result.fired, stimulated)
        return StepTelemetry(
            step=self.step_count - 1,
            sim_time_s=(self.step_count - 1) * cfg.dt,
            n_spikes=result.n_spikes,
            n_background_events=result.n_background_events,
            n_stimulated=result.n_stimulated,
            stimulated_spikes=stimulated_spikes,
            recurrent_l1=float(np.abs(self.last_recurrent).mean()) if self.n else 0.0,
            mean_voltage=float(self.voltage.mean()) if self.n else 0.0,
            population_spikes=pop_spikes,
            population_rates_hz=pop_rates,
            spike_checksum=result.checksum,
            n_outgoing_edges=result.n_outgoing_edges,
        )
