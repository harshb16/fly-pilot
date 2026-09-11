"""Simplified fly-eye encoder: cubemap atlas → R1–R6 currents.

MODELED visual encoding, grounded in the rendered scene:

- Rec.709 luminance
- absolute temporal luminance change (previous frame)
- coarse spatial contrast (center vs ring)
- optional green-opponency term (RGB approximation; MaleCNS R1–R6 are not
  UV-calibrated here)

Coefficients are engineering choices inspired by the documented Fly64 retinal
drive, reimplemented from ``docs/research-notes.md``. They are not measured
photoreceptor physiology.

Each receptor samples a seven-point spherical cone on the cubemap, so two
receptors that look in different directions receive different signals.

Currents are canonicalized through ``float16`` before they reach MaleCNS so a
recorded blob replayed later is bit-identical to the live stimulus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_pilot.brain.vision.geometry import ATLAS_HEIGHT, ATLAS_WIDTH, CameraCalibration
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.brain.vision.scene import atlas_checksum
from fly_pilot.brain.vision.stimulus import RetinalStimulusFrame

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class EncoderConfig:
    """Modeled retinal drive. Not a biophysical fit."""

    luminance_weight: float = 0.45
    temporal_weight: float = 1.6
    contrast_weight: float = 0.35
    green_weight: float = 0.25
    current_scale: float = 0.62
    clip_max: float = 1.0

    def as_dict(self) -> dict:
        return {
            "luminance_weight": self.luminance_weight,
            "temporal_weight": self.temporal_weight,
            "contrast_weight": self.contrast_weight,
            "green_weight": self.green_weight,
            "current_scale": self.current_scale,
            "clip_max": self.clip_max,
            "note": (
                "Modeled encoder. Luminance / |ΔL| / green weights follow the "
                "documented Fly64-style drive; spatial contrast is added here. "
                "current = clip(drive, 0, clip_max) * current_scale."
            ),
        }


class FlyEyeEncoder:
    """Deterministic atlas → RetinalStimulusFrame transform."""

    def __init__(
        self,
        mapping: PhotoreceptorMap,
        config: EncoderConfig | None = None,
        calibration: CameraCalibration | None = None,
    ) -> None:
        self.mapping = mapping
        self.config = config or EncoderConfig()
        self.calibration = calibration or CameraCalibration()
        self._previous_rgb = np.zeros((mapping.n_receptors, 3), dtype=np.float32)
        self._has_previous = False
        n = mapping.n_receptors
        self.last_luminance = np.zeros(n, dtype=np.float32)
        self.last_temporal = np.zeros(n, dtype=np.float32)
        self.last_contrast = np.zeros(n, dtype=np.float32)
        self.last_green = np.zeros(n, dtype=np.float32)

    def reset(self) -> None:
        self._previous_rgb[:] = 0
        self._has_previous = False
        self.last_luminance[:] = 0
        self.last_temporal[:] = 0
        self.last_contrast[:] = 0
        self.last_green[:] = 0

    def sample_rgb(self, atlas: np.ndarray) -> np.ndarray:
        """(N, 3) float32 RGB in 0–1, cone-weighted."""
        if atlas.shape != (ATLAS_HEIGHT, ATLAS_WIDTH, 3):
            raise ValueError(
                f"atlas must be {(ATLAS_HEIGHT, ATLAS_WIDTH, 3)}, got {atlas.shape}"
            )
        flat = atlas.reshape(-1, 3).astype(np.float32) / 255.0
        samples = flat[self.mapping.sample_atlas_indices]  # (N, 7, 3)
        weights = self.mapping.sample_weights[None, :, None]
        return np.sum(samples * weights, axis=1)

    def encode(
        self,
        atlas: np.ndarray,
        *,
        neural_step: int = 0,
        sim_time_s: float = 0.0,
    ) -> RetinalStimulusFrame:
        rgb = self.sample_rgb(atlas)
        lum = rgb @ LUMA
        if self._has_previous:
            prev_lum = self._previous_rgb @ LUMA
            temporal = np.abs(lum - prev_lum)
        else:
            temporal = np.zeros_like(lum)
        # Spatial contrast: |center − mean(ring)| using the same 7 samples.
        flat = atlas.reshape(-1, 3).astype(np.float32) / 255.0
        samples = flat[self.mapping.sample_atlas_indices]
        sample_lum = samples @ LUMA
        center = sample_lum[:, 0]
        ring = sample_lum[:, 1:].mean(axis=1)
        contrast = np.abs(center - ring)
        green = np.maximum(rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2]), 0.0)
        cfg = self.config
        drive = (
            cfg.luminance_weight * lum
            + cfg.temporal_weight * temporal
            + cfg.contrast_weight * contrast
            + cfg.green_weight * green
        )
        drive = np.clip(drive, 0.0, cfg.clip_max)
        # Canonical storage is float16. Round-trip here so live MaleCNS and
        # replay of stored blobs see identical currents.
        currents = (drive * cfg.current_scale).astype(np.float16).astype(np.float32)
        self._previous_rgb = rgb
        self._has_previous = True
        self.last_luminance = lum.astype(np.float32)
        self.last_temporal = temporal.astype(np.float32)
        self.last_contrast = contrast.astype(np.float32)
        self.last_green = green.astype(np.float32)
        return RetinalStimulusFrame(
            currents=currents,
            luminance=lum.astype(np.float32),
            neural_step=neural_step,
            sim_time_s=sim_time_s,
            atlas_sha256=atlas_checksum(atlas),
            extras={
                "mean_temporal": float(temporal.mean()) if lum.size else 0.0,
                "mean_contrast": float(contrast.mean()) if lum.size else 0.0,
                "mean_green": float(green.mean()) if lum.size else 0.0,
            },
        )

    def summary(self, frame: RetinalStimulusFrame) -> dict:
        left = self.mapping.left_mask()
        right = self.mapping.right_mask()
        return {
            "n_receptors": frame.n_receptors,
            "mean_luminance": float(frame.luminance.mean()) if frame.n_receptors else 0.0,
            "mean_current": float(frame.currents.mean()) if frame.n_receptors else 0.0,
            "mean_temporal": float(self.last_temporal.mean()) if frame.n_receptors else 0.0,
            "left_mean_luminance": float(frame.luminance[left].mean()) if left.any() else 0.0,
            "right_mean_luminance": float(frame.luminance[right].mean()) if right.any() else 0.0,
            "current_sha256": frame.current_sha256,
            "atlas_sha256": frame.atlas_sha256,
        }
