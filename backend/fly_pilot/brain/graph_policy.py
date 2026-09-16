"""Trainable MaleCNS-topology graph policy for aircraft control.

This is deliberately separate from the fixed-MaleCNS ``FLY CONTROL`` path.
The policy aggregates the measured MaleCNS connectome into annotated
superclasses, injects aircraft telemetry into sensory populations, propagates
messages over that directed graph, and decodes controls from descending and
motor/efferent populations.  Its weights are task-trained; it is not a
biological simulation and it does not claim that fly neurons naturally encode
Cessna state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from fly_pilot.brain.connectome import Connectome
from fly_pilot.guidance import (
    DEFAULT_AIM_ALONG_M,
    DEFAULT_GLIDESLOPE_DEG,
    glideslope_error_m,
    heading_error_deg,
)
from fly_pilot.runway import Runway
from fly_pilot.state import AircraftControls, AircraftObservation

GRAPH_POLICY_KIND = "trainable_malecns_population_graph_policy"

OBSERVATION_FEATURES = (
    "sim_time_s",
    "along_m",
    "right_m",
    "alt_agl_m",
    "airspeed_kts",
    "groundspeed_kts",
    "vertical_speed_fpm",
    "pitch_deg",
    "roll_deg",
    "heading_error_deg",
    "alpha_deg",
    "beta_deg",
    "p_deg_s",
    "q_deg_s",
    "r_deg_s",
    "glideslope_error_m",
)

SENSORY_SUPERCLASSES = (
    "ol_sensory",
    "cb_sensory",
    "vnc_sensory",
    "sensory_ascending",
    "sensory_descending",
)

READOUT_SUPERCLASSES = (
    "descending_neuron",
    "efferent_descending",
    "vnc_efferent",
    "cb_motor",
    "vnc_motor",
)


def default_graph_checkpoint_path() -> Path:
    return Path("artifacts/connectome_graph/best.pt")


def observation_vector(obs: AircraftObservation, runway: Runway | None = None) -> np.ndarray:
    """Return the declared telemetry vector used by the graph policy."""
    runway = runway or Runway()
    return np.asarray(
        [
            obs.sim_time_s,
            obs.along_m,
            obs.right_m,
            obs.alt_agl_m,
            obs.airspeed_kts,
            obs.groundspeed_kts,
            obs.vertical_speed_fpm,
            obs.pitch_deg,
            obs.roll_deg,
            heading_error_deg(obs.heading_deg, runway.heading_deg),
            obs.alpha_deg,
            obs.beta_deg,
            obs.p_deg_s,
            obs.q_deg_s,
            obs.r_deg_s,
            glideslope_error_m(obs.alt_agl_m, obs.along_m),
        ],
        dtype=np.float32,
    )


def table_observation_matrix(columns: dict[str, np.ndarray], runway: Runway | None = None) -> np.ndarray:
    """Vectorized equivalent of :func:`observation_vector` for Parquet data."""
    runway = runway or Runway()
    heading = np.asarray(columns["heading_deg"], dtype=np.float32)
    heading_error = (runway.heading_deg - heading + 180.0) % 360.0 - 180.0
    along = np.asarray(columns["along_m"], dtype=np.float32)
    alt = np.asarray(columns["alt_agl_m"], dtype=np.float32)
    # Keep this formula aligned with guidance.glideslope_error_m.
    target_alt = np.tan(np.deg2rad(DEFAULT_GLIDESLOPE_DEG)) * np.maximum(
        DEFAULT_AIM_ALONG_M - along,
        30.0,
    )
    target_alt = np.where(along >= DEFAULT_AIM_ALONG_M, np.minimum(target_alt, 0.8), target_alt)
    glide_error = alt - target_alt
    values = {
        "sim_time_s": columns["sim_time_s"],
        "along_m": along,
        "right_m": columns["right_m"],
        "alt_agl_m": alt,
        "airspeed_kts": columns["airspeed_kts"],
        "groundspeed_kts": columns["groundspeed_kts"],
        "vertical_speed_fpm": columns["vertical_speed_fpm"],
        "pitch_deg": columns["pitch_deg"],
        "roll_deg": columns["roll_deg"],
        "heading_error_deg": heading_error,
        "alpha_deg": columns["alpha_deg"],
        "beta_deg": columns["beta_deg"],
        "p_deg_s": columns["p_deg_s"],
        "q_deg_s": columns["q_deg_s"],
        "r_deg_s": columns["r_deg_s"],
        "glideslope_error_m": glide_error,
    }
    return np.column_stack([np.asarray(values[name], dtype=np.float32) for name in OBSERVATION_FEATURES])


@dataclass(frozen=True)
class PopulationGraph:
    names: tuple[str, ...]
    adjacency: np.ndarray
    sensory_mask: np.ndarray
    readout_mask: np.ndarray
    neuron_counts: np.ndarray
    source_manifest: dict[str, Any]

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update("\0".join(self.names).encode())
        for array in (self.adjacency, self.sensory_mask, self.readout_mask, self.neuron_counts):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "adjacency": self.adjacency.astype(float).tolist(),
            "sensory_mask": self.sensory_mask.astype(float).tolist(),
            "readout_mask": self.readout_mask.astype(float).tolist(),
            "neuron_counts": self.neuron_counts.astype(int).tolist(),
            "sha256": self.sha256,
            "source_manifest": self.source_manifest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PopulationGraph":
        graph = cls(
            names=tuple(str(x) for x in data["names"]),
            adjacency=np.asarray(data["adjacency"], dtype=np.float32),
            sensory_mask=np.asarray(data["sensory_mask"], dtype=np.float32),
            readout_mask=np.asarray(data["readout_mask"], dtype=np.float32),
            neuron_counts=np.asarray(data["neuron_counts"], dtype=np.int64),
            source_manifest=dict(data.get("source_manifest") or {}),
        )
        expected = data.get("sha256")
        if expected and expected != graph.sha256:
            raise ValueError("population graph hash mismatch")
        return graph


def build_population_graph(connectome: Connectome) -> PopulationGraph:
    """Aggregate measured neuron edges into a signed superclass graph."""
    names = tuple(sorted({str(x) for x in connectome.superclass if str(x)}))
    lookup = {name: i for i, name in enumerate(names)}
    group = np.asarray([lookup[str(x)] for x in connectome.superclass], dtype=np.int32)
    n_groups = len(names)
    rows = np.arange(connectome.n_neurons, dtype=np.int32)
    membership = sparse.csr_matrix(
        (np.ones(connectome.n_neurons, dtype=np.float32), (rows, group)),
        shape=(connectome.n_neurons, n_groups),
    )
    signed = connectome.counts.tocsr().astype(np.float32, copy=True)
    signed.data *= connectome.sign[signed.indices].astype(np.float32)
    aggregated = (membership.T @ signed @ membership).toarray().astype(np.float32)
    incoming = np.maximum(np.abs(aggregated).sum(axis=1, keepdims=True), 1.0)
    adjacency = aggregated / incoming
    neuron_counts = np.bincount(group, minlength=n_groups).astype(np.int64)
    sensory_mask = np.asarray([name in SENSORY_SUPERCLASSES for name in names], dtype=np.float32)
    readout_mask = np.asarray([name in READOUT_SUPERCLASSES for name in names], dtype=np.float32)
    if not sensory_mask.any() or not readout_mask.any():
        raise ValueError("MaleCNS annotations do not contain required sensory/readout superclasses")
    return PopulationGraph(
        names=names,
        adjacency=adjacency,
        sensory_mask=sensory_mask,
        readout_mask=readout_mask,
        neuron_counts=neuron_counts,
        source_manifest=dict(connectome.manifest),
    )


@dataclass
class GraphPolicyConfig:
    observation_dim: int = len(OBSERVATION_FEATURES)
    node_hidden: int = 64
    temporal_hidden: int = 96
    message_layers: int = 5
    dropout: float = 0.05
    kind: str = GRAPH_POLICY_KIND


class ConnectomeGraphPolicy(nn.Module):
    """Task-trained policy whose message paths follow MaleCNS population edges."""

    def __init__(
        self,
        config: GraphPolicyConfig,
        graph: PopulationGraph,
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.graph = graph
        n = len(graph.names)
        h = config.node_hidden
        self.register_buffer("adjacency", torch.as_tensor(graph.adjacency, dtype=torch.float32))
        self.register_buffer("sensory_mask", torch.as_tensor(graph.sensory_mask, dtype=torch.float32).view(1, n, 1))
        readout = graph.readout_mask / max(float(graph.readout_mask.sum()), 1.0)
        self.register_buffer("readout_mask", torch.as_tensor(readout, dtype=torch.float32).view(1, n, 1))
        self.register_buffer("mean", torch.zeros(config.observation_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(config.observation_dim) if std is None else torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6))
        self.node_embedding = nn.Parameter(torch.empty(n, h))
        nn.init.normal_(self.node_embedding, std=0.03)
        self.observation_encoder = nn.Sequential(
            nn.Linear(config.observation_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.self_layers = nn.ModuleList(nn.Linear(h, h) for _ in range(config.message_layers))
        self.message_layers = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(config.message_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(config.message_layers))
        self.dropout = nn.Dropout(config.dropout)
        self.temporal = nn.GRU(h, config.temporal_hidden, batch_first=True)
        self.control_head = nn.Linear(config.temporal_hidden, 4)
        self._hidden: torch.Tensor | None = None

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def reset_state(self) -> None:
        self._hidden = None

    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6))

    def _graph_encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        encoded = self.observation_encoder((x - self.mean) / self.std).reshape(batch * steps, 1, -1)
        nodes = self.node_embedding.reshape(1, len(self.graph.names), -1).expand(batch * steps, -1, -1)
        nodes = nodes + self.sensory_mask * encoded
        for self_layer, message_layer, norm in zip(self.self_layers, self.message_layers, self.norms):
            message = torch.einsum("ij,bjh->bih", self.adjacency, nodes)
            update = torch.nn.functional.gelu(self_layer(nodes) + message_layer(message))
            nodes = norm(nodes + self.dropout(update))
        pooled = (nodes * self.readout_mask).sum(dim=1)
        return pooled.reshape(batch, steps, -1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        graph_features = self._graph_encode(x)
        temporal, hidden_out = self.temporal(graph_features, hidden)
        raw = self.control_head(temporal)
        controls = torch.cat((torch.tanh(raw[..., :3]), torch.sigmoid(raw[..., 3:4])), dim=-1)
        return controls, hidden_out

    def step_numpy(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(features, dtype=np.float32).reshape(1, 1, -1))
        self.eval()
        with torch.no_grad():
            y, self._hidden = self.forward(x, self._hidden)
        return y.reshape(-1).cpu().numpy().astype(np.float32)


@dataclass
class GraphPolicyArtifact:
    config: GraphPolicyConfig
    graph: PopulationGraph
    model: ConnectomeGraphPolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": GRAPH_POLICY_KIND,
            "biological_learning": False,
            "uses_aircraft_telemetry": True,
            "config": asdict(self.config),
            "graph": self.graph.as_dict(),
            "state_dict": self.model.state_dict(),
            "observation_features": list(OBSERVATION_FEATURES),
            "metadata": self.metadata,
        }
        torch.save(payload, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = {
            "kind": GRAPH_POLICY_KIND,
            "checkpoint_sha256": digest,
            "population_graph_sha256": self.graph.sha256,
            "population_names": list(self.graph.names),
            "sensory_populations": [n for n, m in zip(self.graph.names, self.graph.sensory_mask) if m],
            "readout_populations": [n for n, m in zip(self.graph.names, self.graph.readout_mask) if m],
            "observation_features": list(OBSERVATION_FEATURES),
            "parameter_count": self.model.parameter_count(),
            "uses_aircraft_telemetry": True,
            "biological_learning": False,
            "scientific_scope": (
                "Measured MaleCNS connectivity is aggregated to population topology; "
                "all policy weights are task-trained. This is not a fixed-brain simulation."
            ),
            **self.metadata,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "GraphPolicyArtifact":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("kind") != GRAPH_POLICY_KIND:
            raise ValueError(f"not a {GRAPH_POLICY_KIND} checkpoint")
        if tuple(payload.get("observation_features") or ()) != OBSERVATION_FEATURES:
            raise ValueError("graph policy observation feature order mismatch")
        config = GraphPolicyConfig(**payload["config"])
        graph = PopulationGraph.from_dict(payload["graph"])
        model = ConnectomeGraphPolicy(config, graph)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(config=config, graph=graph, model=model, metadata=dict(payload.get("metadata") or {}))


def vector_to_controls(values: np.ndarray) -> AircraftControls:
    return AircraftControls(
        aileron=float(values[0]),
        elevator=float(values[1]),
        rudder=float(values[2]),
        throttle=float(values[3]),
    ).clamped()
