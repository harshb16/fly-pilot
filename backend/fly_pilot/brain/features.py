"""Descending-neuron and visual-pathway spike-rate features.

Identifies candidate motor-side readouts from MaleCNS annotations. This
milestone records features only. It does **not** map neurons onto aileron /
elevator / rudder / throttle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.telemetry import population_spike_count, rate_hz


def _side_of(connectome: Connectome, index: int) -> str:
    side = str(connectome.side[index]).strip().upper()
    inst = str(connectome.instance[index]).strip().upper()
    if side in {"L", "LEFT"} or inst.endswith("_L") or "(L)" in inst:
        return "L"
    if side in {"R", "RIGHT"} or inst.endswith("_R") or "(R)" in inst:
        return "R"
    return "?"


@dataclass
class PopulationSpec:
    name: str
    indices: np.ndarray
    body_ids: np.ndarray
    types: np.ndarray
    sides: np.ndarray

    @property
    def n(self) -> int:
        return int(self.indices.size)

    def left_indices(self) -> np.ndarray:
        return self.indices[self.sides == "L"]

    def right_indices(self) -> np.ndarray:
        return self.indices[self.sides == "R"]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "n_left": int((self.sides == "L").sum()),
            "n_right": int((self.sides == "R").sum()),
            "n_unknown_side": int((self.sides == "?").sum()),
            "unique_types": int(len({str(t) for t in self.types})),
        }


def _spec(connectome: Connectome, name: str, indices: np.ndarray) -> PopulationSpec:
    indices = np.asarray(indices, dtype=np.int32)
    sides = np.array([_side_of(connectome, int(i)) for i in indices], dtype=object)
    types = np.array(
        [str(connectome.type[i] or connectome.flywire_type[i]) for i in indices],
        dtype=object,
    )
    return PopulationSpec(
        name=name,
        indices=indices,
        body_ids=connectome.body_ids[indices] if indices.size else np.zeros(0, dtype=np.int64),
        types=types,
        sides=sides,
    )


def descending_population(connectome: Connectome) -> PopulationSpec:
    index = NeuronIndex(connectome)
    return _spec(connectome, "descending_neuron", index.query(superclass="descending_neuron"))


def visual_debug_populations(connectome: Connectome) -> dict[str, PopulationSpec]:
    """Selected visual-pathway groups for diagnostics (not a decoder)."""
    index = NeuronIndex(connectome)
    wanted = [
        ("R1-R6", dict(cell_type="R1-R6")),
        ("L1", dict(type="L1")),
        ("L2", dict(type="L2")),
        ("L3", dict(type="L3")),
        ("visual_projection", dict(superclass="visual_projection")),
        ("DNp01", dict(cell_type="DNp01")),
        ("DNa02", dict(cell_type="DNa02")),
        ("DNg13", dict(cell_type="DNg13")),
        ("DNg100", dict(cell_type="DNg100")),
    ]
    out: dict[str, PopulationSpec] = {}
    for name, filters in wanted:
        ids = index.query(**filters)
        if ids.size:
            out[name] = _spec(connectome, name, ids)
    return out


class DescendingNeuronFeatureExtractor:
    """Rolling spike-rate features over configurable windows.

    Default windows: 5 steps (100 ms) and 13 steps (260 ms) at dt = 20 ms.
    Left/right rates are retained from annotations. Per-neuron rates use the
    longest window and are the candidate decoder vector for a later milestone.
    """

    def __init__(
        self,
        spec: PopulationSpec,
        *,
        dt: float = 0.020,
        windows: tuple[int, ...] = (5, 13),
    ) -> None:
        if not windows:
            raise ValueError("at least one spike-rate window is required")
        self.spec = spec
        self.dt = float(dt)
        self.windows = tuple(sorted(int(w) for w in windows))
        self.max_window = max(self.windows)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.max_window)
        self.last_rates: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._buffer.clear()
        self.last_rates = {}

    def update(self, fired: np.ndarray) -> dict[str, float | np.ndarray | int]:
        if self.spec.n == 0:
            return {
                "n_descending": 0,
                "spikes_this_step": 0,
            }
        spikes = fired[self.spec.indices].astype(np.uint8, copy=False)
        self._buffer.append(spikes.copy())
        stacked = np.stack(tuple(self._buffer), axis=0)
        features: dict[str, float | np.ndarray | int] = {
            "n_descending": self.spec.n,
            "spikes_this_step": int(spikes.sum()),
        }
        left = self.spec.sides == "L"
        right = self.spec.sides == "R"
        for window in self.windows:
            recent = stacked[-window:]
            counts = recent.sum(axis=0)
            duration = window * self.dt
            rates = counts.astype(np.float32) / duration
            self.last_rates[window] = rates
            features[f"mean_hz_w{window}"] = float(rates.mean())
            if left.any():
                features[f"left_mean_hz_w{window}"] = float(rates[left].mean())
            if right.any():
                features[f"right_mean_hz_w{window}"] = float(rates[right].mean())
            features[f"rates_w{window}"] = rates
        return features

    def feature_vector(self) -> np.ndarray:
        """Per-neuron rate over the longest window (candidate decoder input)."""
        rates = self.last_rates.get(self.max_window)
        if rates is None:
            return np.zeros(self.spec.n, dtype=np.float32)
        return rates

    def compact_summary(self) -> dict[str, float | int]:
        vec = self.feature_vector()
        left = self.spec.sides == "L"
        right = self.spec.sides == "R"
        return {
            "n_descending": self.spec.n,
            "n_left": int(left.sum()),
            "n_right": int(right.sum()),
            "mean_hz": float(vec.mean()) if vec.size else 0.0,
            "left_mean_hz": float(vec[left].mean()) if left.any() else 0.0,
            "right_mean_hz": float(vec[right].mean()) if right.any() else 0.0,
            "max_hz": float(vec.max()) if vec.size else 0.0,
            "window_steps": self.max_window,
        }


class VisualPopulationTracker:
    """Instantaneous and EMA rates for selected visual pathway groups."""

    def __init__(self, populations: dict[str, PopulationSpec], *, dt: float = 0.020, ema: float = 0.8) -> None:
        self.populations = populations
        self.dt = dt
        self.ema = ema
        self._ema_hz: dict[str, float] = {name: 0.0 for name in populations}

    def reset(self) -> None:
        self._ema_hz = {name: 0.0 for name in self.populations}

    def update(self, fired: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, spec in self.populations.items():
            count = population_spike_count(fired, spec.indices)
            inst = rate_hz(count, spec.n, self.dt)
            self._ema_hz[name] = self.ema * self._ema_hz[name] + (1.0 - self.ema) * inst
            out[f"{name}_hz"] = inst
            out[f"{name}_ema_hz"] = self._ema_hz[name]
            out[f"{name}_spikes"] = float(count)
        return out
