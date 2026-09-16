"""External causal temporal decoder: DN rates → Cessna inceptors.

This is **not** biological synaptic learning. MaleCNS weights stay frozen.
The learned object is a small PyTorch GRU that reads descending-neuron
spike-rate features and emits aileron / elevator / rudder / throttle.

Decoder input is exclusively MaleCNS-derived DN activity. Aircraft
telemetry must never be concatenated into the feature vector.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fly_pilot.state import AircraftControls

CONTROL_NAMES = ("aileron", "elevator", "rudder", "throttle")
DECODER_KIND = "fixed_malecns_plus_trained_temporal_decoder"
DECODER_INPUT_KIND = "dn_windowed_rates"
TARGET_CONVENTIONS = {
    "aileron": "[-1, 1] positive rolls right",
    "elevator": "[-1, 1] positive is stick-back / nose-up (pilot convention)",
    "rudder": "[-1, 1] positive yaws right",
    "throttle": "[0, 1] 0 idle, 1 full",
    "source": "ExpertLandingController.act() at the same neural timestep",
    "not_biological": True,
}


@dataclass
class DecoderConfig:
    n_descending: int = 1314
    windows: tuple[int, ...] = (5, 13)
    hidden_linear: int = 256
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 50
    dt: float = 0.020
    kind: str = DECODER_KIND
    input_kind: str = DECODER_INPUT_KIND

    @property
    def input_dim(self) -> int:
        return int(self.n_descending * len(self.windows))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windows"] = list(self.windows)
        payload["input_dim"] = self.input_dim
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecoderConfig":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        if "windows" in known:
            known["windows"] = tuple(int(w) for w in known["windows"])
        return cls(**known)


class CausalTemporalDecoder(nn.Module):
    """Linear → LayerNorm → GELU → 2-layer GRU → four control heads.

    Causal: each neural timestep uses only current and past hidden state.
    """

    def __init__(self, config: DecoderConfig, *, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        super().__init__()
        self.config = config
        dim = config.input_dim
        mean_t = torch.zeros(dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.ones(dim) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", torch.clamp(std_t, min=1e-6))
        self.fc = nn.Linear(dim, config.hidden_linear)
        self.norm = nn.LayerNorm(config.hidden_linear)
        self.act = nn.GELU()
        self.gru = nn.GRU(
            input_size=config.hidden_linear,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_aileron = nn.Linear(config.gru_hidden, 1)
        self.head_elevator = nn.Linear(config.gru_hidden, 1)
        self.head_rudder = nn.Linear(config.gru_hidden, 1)
        self.head_throttle = nn.Linear(config.gru_hidden, 1)
        self._hidden: torch.Tensor | None = None

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        dim = self.config.input_dim
        mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32).reshape(dim), dtype=torch.float32)
        std_t = torch.as_tensor(np.asarray(std, dtype=np.float32).reshape(dim), dtype=torch.float32)
        self.mean.copy_(mean_t)
        self.std.copy_(torch.clamp(std_t, min=1e-6))

    def reset_state(self) -> None:
        self._hidden = None

    def hidden_summary(self) -> dict[str, float]:
        if self._hidden is None:
            return {"gru_hidden_norm": 0.0, "gru_hidden_mean": 0.0, "gru_hidden_max": 0.0}
        h = self._hidden.detach()
        return {
            "gru_hidden_norm": float(torch.linalg.vector_norm(h).item()),
            "gru_hidden_mean": float(h.mean().item()),
            "gru_hidden_max": float(h.abs().max().item()),
        }

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, F) DN rates only. Returns y (B, T, 4) and GRU hidden."""
        z = self.act(self.norm(self.fc(self._normalize(x))))
        out, hidden_out = self.gru(z, hidden)
        aileron = torch.tanh(self.head_aileron(out))
        elevator = torch.tanh(self.head_elevator(out))
        rudder = torch.tanh(self.head_rudder(out))
        throttle = torch.sigmoid(self.head_throttle(out))
        y = torch.cat([aileron, elevator, rudder, throttle], dim=-1)
        return y, hidden_out

    def step_numpy(self, features: np.ndarray) -> np.ndarray:
        """One causal neural timestep. Updates persistent GRU state."""
        vec = np.asarray(features, dtype=np.float32).reshape(1, 1, -1)
        if vec.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"decoder expected {self.config.input_dim} DN features, got {vec.shape[-1]}"
            )
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(vec)
            y, self._hidden = self.forward(x, self._hidden)
        return y.view(-1).cpu().numpy().astype(np.float32)

    def controls_from_features(self, features: np.ndarray) -> AircraftControls:
        y = self.step_numpy(features)
        return vector_to_controls(y)


def vector_to_controls(y: np.ndarray) -> AircraftControls:
    return AircraftControls(
        aileron=float(y[0]),
        elevator=float(y[1]),
        rudder=float(y[2]),
        throttle=float(y[3]),
    ).clamped()


def controls_to_vector(controls: AircraftControls) -> np.ndarray:
    return np.array(
        [controls.aileron, controls.elevator, controls.rudder, controls.throttle],
        dtype=np.float32,
    )


@dataclass
class DecoderArtifact:
    """Portable trained-decoder bundle. Weights are frozen MaleCNS + this GRU."""

    config: DecoderConfig
    model: CausalTemporalDecoder
    dn_body_ids: np.ndarray
    git_commit: str = ""
    git_dirty: bool = False
    training_seed: int = 0
    training_command: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)
    seed_ranges: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    target_conventions: dict[str, Any] = field(default_factory=lambda: dict(TARGET_CONVENTIONS))
    biological_learning: bool = False
    kind: str = DECODER_KIND

    def assert_dn_ordering(self, body_ids: np.ndarray) -> None:
        got = np.asarray(body_ids, dtype=np.int64)
        expected = np.asarray(self.dn_body_ids, dtype=np.int64)
        if got.shape != expected.shape or not np.array_equal(got, expected):
            raise ValueError(
                "DN body-id ordering does not match the training artifact. "
                "Refusing to decode with a misaligned MaleCNS population."
            )

    def metadata(self, *, checkpoint_sha256: str | None = None) -> dict[str, Any]:
        mean = self.model.mean.detach().cpu().numpy()
        std = self.model.std.detach().cpu().numpy()
        return {
            "kind": self.kind,
            "biological_learning": False,
            "input_kind": DECODER_INPUT_KIND,
            "input_fields": ["dn_rates_100ms", "dn_rates_260ms"],
            "excludes_aircraft_telemetry": True,
            "config": self.config.as_dict(),
            "parameter_count": self.model.parameter_count(),
            "dn_n": int(self.dn_body_ids.size),
            "dn_body_ids": [int(x) for x in self.dn_body_ids.tolist()],
            "scaler": {
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
            },
            "target_conventions": self.target_conventions,
            "dataset": self.dataset,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "training_seed": self.training_seed,
            "training_command": self.training_command,
            "dependencies": self.dependencies,
            "seed_ranges": self.seed_ranges,
            "metrics": self.metrics,
            "checkpoint_sha256": checkpoint_sha256,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "biological_learning": False,
            "config": self.config.as_dict(),
            "state_dict": self.model.state_dict(),
            "dn_body_ids": np.asarray(self.dn_body_ids, dtype=np.int64),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "training_seed": int(self.training_seed),
            "training_command": list(self.training_command),
            "dependencies": dict(self.dependencies),
            "seed_ranges": dict(self.seed_ranges),
            "dataset": self.dataset,
            "metrics": self.metrics,
            "target_conventions": self.target_conventions,
            "input_kind": DECODER_INPUT_KIND,
        }
        torch.save(payload, path)
        checkpoint_sha256 = sha256_path(path)
        sidecar = path.with_suffix(".meta.json")
        meta = self.metadata(checkpoint_sha256=checkpoint_sha256)
        # Body ids and scaler live in the .pt file; keep the JSON inspectable.
        meta["dn_body_ids"] = meta["dn_body_ids"][:12] + ["..."] if len(meta["dn_body_ids"]) > 12 else meta["dn_body_ids"]
        meta["scaler"] = {
            "mean_preview": meta["scaler"]["mean"][:8],
            "std_preview": meta["scaler"]["std"][:8],
            "n": int(self.config.input_dim),
        }
        sidecar.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "DecoderArtifact":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"decoder checkpoint not found: {path}. Restore the committed "
                "artifact or set FLYPILOT_DECODER_PATH."
            )
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = DecoderConfig.from_dict(payload["config"])
        model = CausalTemporalDecoder(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(
            config=config,
            model=model,
            dn_body_ids=np.asarray(payload["dn_body_ids"], dtype=np.int64),
            git_commit=str(payload.get("git_commit", "")),
            git_dirty=bool(payload.get("git_dirty", False)),
            training_seed=int(payload.get("training_seed", 0)),
            training_command=list(payload.get("training_command") or []),
            dependencies=dict(payload.get("dependencies") or {}),
            seed_ranges=dict(payload.get("seed_ranges") or {}),
            dataset=dict(payload.get("dataset") or {}),
            metrics=dict(payload.get("metrics") or {}),
            target_conventions=dict(payload.get("target_conventions") or TARGET_CONVENTIONS),
            kind=str(payload.get("kind", DECODER_KIND)),
        )


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def default_checkpoint_path() -> Path:
    import os

    override = os.environ.get("FLYPILOT_DECODER_PATH")
    if override:
        return Path(override)
    return Path("artifacts/decoder/best.pt")
