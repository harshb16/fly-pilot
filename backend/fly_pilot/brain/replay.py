"""Deterministic replay helpers for MaleCNS LIF runs.

Two runs with the same prepared connectome, LIFConfig (including seed), and
stimulation sequence must produce identical spike checksums on the same
platform / NumPy / SciPy versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def spike_checksum(fired: np.ndarray) -> str:
    """Checksum of a boolean or 0/1 spike vector."""
    payload = np.ascontiguousarray(fired.astype(np.uint8, copy=False))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def chain_checksum(previous: str, fired: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(previous.encode("ascii"))
    digest.update(bytes.fromhex(spike_checksum(fired)))
    return digest.hexdigest()


@dataclass
class SpikeTrace:
    seed: int
    n_neurons: int
    n_steps: int
    dt: float
    stimulation: dict[str, Any]
    step_checksums: list[str] = field(default_factory=list)
    step_spike_counts: list[int] = field(default_factory=list)
    final_checksum: str = ""
    notes: str = (
        "Checksum of boolean spike vectors. Reproducible on the same platform "
        "with the same prepared connectome, LIFConfig, and stimulation."
    )

    def record(self, fired: np.ndarray) -> str:
        checksum = spike_checksum(fired)
        self.step_checksums.append(checksum)
        self.step_spike_counts.append(int(np.count_nonzero(fired)))
        self.final_checksum = (
            checksum if not self.final_checksum else chain_checksum(self.final_checksum, fired)
        )
        return checksum

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def read(cls, path: Path) -> "SpikeTrace":
        payload = json.loads(path.read_text())
        return cls(**payload)


def traces_match(a: SpikeTrace, b: SpikeTrace) -> bool:
    return (
        a.n_neurons == b.n_neurons
        and a.n_steps == b.n_steps
        and a.seed == b.seed
        and a.final_checksum == b.final_checksum
        and a.step_checksums == b.step_checksums
        and a.step_spike_counts == b.step_spike_counts
    )
