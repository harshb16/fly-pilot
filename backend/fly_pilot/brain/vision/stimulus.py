"""Canonical retinal stimulus frames.

This is the sensory object MaleCNS actually receives. A recorded sequence of
these frames can be replayed without Three.js or JSBSim.

Schema ``retinal-v1``:

- float32 currents, one per mapped R1–R6 receptor (same order as PhotoreceptorMap)
- float32 luminance (0–1) for diagnostics
- SHA-256 of currents and of the source cubemap atlas
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RETINAL_SCHEMA = "retinal-v1"


def _sha256_f32(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values.astype(np.float32, copy=False))
    return hashlib.sha256(payload.tobytes()).hexdigest()


@dataclass
class RetinalStimulusFrame:
    currents: np.ndarray
    luminance: np.ndarray
    neural_step: int = 0
    sim_time_s: float = 0.0
    atlas_sha256: str = ""
    schema: str = RETINAL_SCHEMA
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.currents = np.asarray(self.currents, dtype=np.float32)
        self.luminance = np.asarray(self.luminance, dtype=np.float32)
        if self.currents.shape != self.luminance.shape:
            raise ValueError("currents and luminance must have the same shape")

    @property
    def n_receptors(self) -> int:
        return int(self.currents.size)

    @property
    def current_sha256(self) -> str:
        return _sha256_f32(self.currents)

    def to_bytes(self) -> bytes:
        """Compact canonical payload (schema + step + currents)."""
        header = json.dumps(
            {
                "schema": self.schema,
                "neural_step": int(self.neural_step),
                "sim_time_s": float(self.sim_time_s),
                "n": self.n_receptors,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return header + b"\n" + np.ascontiguousarray(self.currents, dtype=np.float32).tobytes()

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "neural_step": int(self.neural_step),
            "sim_time_s": float(self.sim_time_s),
            "n_receptors": self.n_receptors,
            "current_sha256": self.current_sha256,
            "atlas_sha256": self.atlas_sha256,
            "mean_luminance": float(self.luminance.mean()) if self.n_receptors else 0.0,
            "mean_current": float(self.currents.mean()) if self.n_receptors else 0.0,
            "currents_f16": np.ascontiguousarray(self.currents.astype(np.float16)).tobytes(),
            "luminance_f16": np.ascontiguousarray(self.luminance.astype(np.float16)).tobytes(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RetinalStimulusFrame":
        n = int(record["n_receptors"])
        currents = np.frombuffer(record["currents_f16"], dtype=np.float16).astype(np.float32)
        luminance = np.frombuffer(record["luminance_f16"], dtype=np.float16).astype(np.float32)
        if currents.size != n or luminance.size != n:
            raise ValueError("retinal blob length does not match n_receptors")
        frame = cls(
            currents=currents,
            luminance=luminance,
            neural_step=int(record.get("neural_step", 0)),
            sim_time_s=float(record.get("sim_time_s", 0.0)),
            atlas_sha256=str(record.get("atlas_sha256", "")),
            schema=str(record.get("schema", RETINAL_SCHEMA)),
        )
        stored = record.get("current_sha256")
        if stored and stored != frame.current_sha256:
            # float16 round-trip: recompute is expected to differ slightly.
            # Keep the stored checksum as extras for diagnostics.
            frame.extras["stored_current_sha256"] = stored
        return frame

    @classmethod
    def from_currents(
        cls,
        currents: np.ndarray,
        *,
        luminance: np.ndarray | None = None,
        neural_step: int = 0,
        sim_time_s: float = 0.0,
        atlas_sha256: str = "",
    ) -> "RetinalStimulusFrame":
        currents = np.asarray(currents, dtype=np.float32)
        if luminance is None:
            luminance = np.clip(currents, 0.0, 1.0)
        return cls(
            currents=currents,
            luminance=np.asarray(luminance, dtype=np.float32),
            neural_step=neural_step,
            sim_time_s=sim_time_s,
            atlas_sha256=atlas_sha256,
        )
