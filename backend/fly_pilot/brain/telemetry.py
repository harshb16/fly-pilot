"""Per-step spike and population-rate summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepTelemetry:
    step: int
    sim_time_s: float
    n_spikes: int
    n_background_events: int
    n_stimulated: int
    stimulated_spikes: int
    recurrent_l1: float
    mean_voltage: float
    population_spikes: dict[str, int] = field(default_factory=dict)
    population_rates_hz: dict[str, float] = field(default_factory=dict)
    spike_checksum: str = ""


def population_spike_count(fired: np.ndarray, indices: np.ndarray) -> int:
    if indices.size == 0:
        return 0
    return int(np.count_nonzero(fired[indices]))


def rate_hz(spike_count: int, n_neurons: int, dt: float) -> float:
    if n_neurons <= 0 or dt <= 0:
        return 0.0
    return float(spike_count) / (float(n_neurons) * dt)
