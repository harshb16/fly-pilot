"""Deterministic R1–R6 → visual-direction assignment.

The prepared MaleCNS tables do **not** include measured ommatidial pointing
vectors (the optic-column Excel workbook is intentionally not downloaded).
This module therefore:

1. selects the actual R1–R6 neurons present in the connectome annotations,
2. splits them by annotated side (L/R),
3. sorts each eye by ``body_id``,
4. stretches that order over a documented left/right spherical field.

The spatial layout is reproducible given the same neuron table. It is an
approximation inspired by Fly64/NeuroMechFly field bounds, not recovered
Drosophila optics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.vision.geometry import (
    ELEVATION_LIMIT_DEG,
    LEFT_AZIMUTH,
    RIGHT_AZIMUTH,
    angular_rays,
    cone_rays,
    cone_weights,
    atlas_indices_for_rays,
)


def _side_label(side: str, instance: str) -> str:
    s = str(side).strip().upper()
    inst = str(instance).strip().upper()
    if s in {"L", "LEFT"} or inst.endswith("_L") or "(L)" in inst:
        return "L"
    if s in {"R", "RIGHT"} or inst.endswith("_R") or "(R)" in inst:
        return "R"
    return "?"


def _grid_angles(n: int, az_range: tuple[float, float], el_limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-major spherical grid covering the eye. Deterministic for a given n."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    az_span = az_range[1] - az_range[0]
    el_span = 2.0 * el_limit
    aspect = max(az_span / max(el_span, 1e-6), 0.25)
    cols = max(int(round(np.sqrt(n * aspect))), 1)
    rows = int(np.ceil(n / cols))
    azimuth = np.empty(n, dtype=np.float32)
    elevation = np.empty(n, dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        azimuth[i] = az_range[0] + (c + 0.5) / cols * az_span
        elevation[i] = el_limit - (r + 0.5) / rows * el_span
    return azimuth, elevation


@dataclass
class PhotoreceptorMap:
    """One visual direction per modeled R1–R6 cell."""

    indices: np.ndarray
    body_ids: np.ndarray
    sides: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    rays_local: np.ndarray
    sample_atlas_indices: np.ndarray
    sample_weights: np.ndarray
    n_unknown_side: int = 0

    @property
    def n_receptors(self) -> int:
        return int(self.indices.size)

    def left_mask(self) -> np.ndarray:
        return self.sides == "L"

    def right_mask(self) -> np.ndarray:
        return self.sides == "R"

    def as_dict(self) -> dict:
        return {
            "n_receptors": self.n_receptors,
            "n_left": int(self.left_mask().sum()),
            "n_right": int(self.right_mask().sum()),
            "n_unknown_side": int(self.n_unknown_side),
            "azimuth_range_left_deg": list(LEFT_AZIMUTH),
            "azimuth_range_right_deg": list(RIGHT_AZIMUTH),
            "elevation_limit_deg": ELEVATION_LIMIT_DEG,
            "layout": "body_id rank within eye → regular spherical grid",
        }

    @classmethod
    def synthetic(cls, n_left: int = 16, n_right: int = 16, start_index: int = 0) -> "PhotoreceptorMap":
        """Grid of fake receptors for unit tests (no MaleCNS table required)."""
        n = n_left + n_right
        indices = np.arange(start_index, start_index + n, dtype=np.int32)
        body_ids = np.arange(1000, 1000 + n, dtype=np.int64)
        sides = np.array(["L"] * n_left + ["R"] * n_right, dtype=object)
        az_l, el_l = _grid_angles(n_left, LEFT_AZIMUTH, ELEVATION_LIMIT_DEG)
        az_r, el_r = _grid_angles(n_right, RIGHT_AZIMUTH, ELEVATION_LIMIT_DEG)
        azimuth = np.concatenate((az_l, az_r)).astype(np.float32)
        elevation = np.concatenate((el_l, el_r)).astype(np.float32)
        return cls._from_angles(indices, body_ids, sides, azimuth, elevation, n_unknown_side=0)

    @classmethod
    def from_connectome(cls, connectome: Connectome) -> "PhotoreceptorMap":
        index = NeuronIndex(connectome)
        ids = index.query(cell_type="R1-R6")
        if ids.size == 0:
            raise RuntimeError("no R1-R6 photoreceptors in the prepared connectome")
        sides_raw = [_side_label(connectome.side[i], connectome.instance[i]) for i in ids]
        sides = np.asarray(sides_raw, dtype=object)
        unknown = sides == "?"
        n_unknown = int(unknown.sum())
        # Deterministic leftover assignment: even body_id → L, odd → R.
        if n_unknown:
            body = connectome.body_ids[ids]
            sides[unknown] = np.where(body[unknown] % 2 == 0, "L", "R")
        left = ids[sides == "L"]
        right = ids[sides == "R"]
        left = left[np.argsort(connectome.body_ids[left])]
        right = right[np.argsort(connectome.body_ids[right])]
        az_l, el_l = _grid_angles(int(left.size), LEFT_AZIMUTH, ELEVATION_LIMIT_DEG)
        az_r, el_r = _grid_angles(int(right.size), RIGHT_AZIMUTH, ELEVATION_LIMIT_DEG)
        indices = np.concatenate((left, right)).astype(np.int32)
        body_ids = connectome.body_ids[indices]
        sides_out = np.array(["L"] * left.size + ["R"] * right.size, dtype=object)
        azimuth = np.concatenate((az_l, az_r)).astype(np.float32)
        elevation = np.concatenate((el_l, el_r)).astype(np.float32)
        return cls._from_angles(indices, body_ids, sides_out, azimuth, elevation, n_unknown_side=n_unknown)

    @classmethod
    def _from_angles(
        cls,
        indices: np.ndarray,
        body_ids: np.ndarray,
        sides: np.ndarray,
        azimuth: np.ndarray,
        elevation: np.ndarray,
        n_unknown_side: int,
    ) -> "PhotoreceptorMap":
        rays = angular_rays(azimuth, elevation)
        cone = cone_rays(rays)
        flat_idx = atlas_indices_for_rays(cone.reshape(-1, 3)).reshape(cone.shape[:2])
        return cls(
            indices=np.asarray(indices, dtype=np.int32),
            body_ids=np.asarray(body_ids, dtype=np.int64),
            sides=np.asarray(sides, dtype=object),
            azimuth_deg=np.asarray(azimuth, dtype=np.float32),
            elevation_deg=np.asarray(elevation, dtype=np.float32),
            rays_local=rays,
            sample_atlas_indices=flat_idx,
            sample_weights=cone_weights(),
            n_unknown_side=n_unknown_side,
        )
