"""Fly-view camera geometry and cubemap atlas layout.

MODELED, not measured MaleCNS optics. Angular bounds are inspired by the
wide-field compound-eye coverage summarized from NeuroMechFly v2 (~270°
horizontal, ~17° binocular overlap, ±72° elevation) and documented in
``docs/research-notes.md``. They are not a FlyGym import and not a copy of
unlicensed Fly64 source.

Visual body frame (matches Three.js aircraft local axes after pose):

- +X right
- +Y up
- −Z forward (nose)

World ENU: east, north, up. Three.js world is (east, up, −north).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cubemap face size. 32² × 6 is compact to serialize and cheap to rasterize.
FACE_SIZE = 32
FACE_ORDER = ("forward", "right", "back", "left", "up", "down")
ATLAS_COLS = 3
ATLAS_ROWS = 2
ATLAS_WIDTH = FACE_SIZE * ATLAS_COLS
ATLAS_HEIGHT = FACE_SIZE * ATLAS_ROWS

# NeuroMechFly-inspired wide field (documented approximation).
HORIZONTAL_FOV_DEG = 270.0
OVERLAP_DEG = 17.0
ELEVATION_LIMIT_DEG = 72.0
LEFT_AZIMUTH = (-HORIZONTAL_FOV_DEG / 2.0, OVERLAP_DEG / 2.0)  # −135 … +8.5
RIGHT_AZIMUTH = (-OVERLAP_DEG / 2.0, HORIZONTAL_FOV_DEG / 2.0)  # −8.5 … +135

# Seven-sample acceptance cone (engineering, ~2°).
ACCEPTANCE_SIGMA_DEG = 2.0

# Cockpit-like eye origin: slightly above the JSBSim reference point.
EYE_UP_M = 1.2
EYE_FORWARD_M = 2.4

# Per-eye yaw of the *preview* cameras (not the cubemap). Centers of each eye.
LEFT_EYE_YAW_DEG = -63.25
RIGHT_EYE_YAW_DEG = 63.25
PREVIEW_FOV_DEG = 140.0
PREVIEW_WIDTH = 48
PREVIEW_HEIGHT = 32


# Face forward / right / up in the aircraft visual frame.
# Written from the axis convention above; not vendored from another project.
_FACE_BASES = {
    "forward": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "right": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "back": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


def face_bases() -> np.ndarray:
    """Return (6, 3, 3) float32: per face rows [right, up, forward]."""
    rows = []
    for name in FACE_ORDER:
        right, up, forward = _FACE_BASES[name]
        rows.append([right, up, forward])
    return np.asarray(rows, dtype=np.float32)


def atlas_origin(face_index: int) -> tuple[int, int]:
    col = face_index % ATLAS_COLS
    row = face_index // ATLAS_COLS
    return col * FACE_SIZE, row * FACE_SIZE


def angular_rays(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Unit rays in the aircraft visual frame from azimuth/elevation degrees.

    Azimuth 0 = nose, positive = right. Elevation 0 = horizon, positive = up.
    """
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    ce = np.cos(el)
    x = np.sin(az) * ce  # right
    y = np.sin(el)  # up
    z = -np.cos(az) * ce  # forward is −Z
    rays = np.stack((x, y, z), axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return (rays / np.maximum(norms, 1e-12)).astype(np.float32)


def aircraft_rotation_visual(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """3×3 rotation: visual-local → Three.js world (east, up, −north).

    Matches ``applyJsbsimPose``: Euler order YXZ with y=−heading, x=pitch, z=−roll.
    """
    hy = np.deg2rad(-heading_deg)
    px = np.deg2rad(pitch_deg)
    rz = np.deg2rad(-roll_deg)
    cy, sy = np.cos(hy), np.sin(hy)
    cx, sx = np.cos(px), np.sin(px)
    cz, sz = np.cos(rz), np.sin(rz)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (ry @ rx @ rz_m).astype(np.float32)


def visual_world_to_enu(vec: np.ndarray) -> np.ndarray:
    """Three.js world (x east, y up, z south) → ENU (east, north, up)."""
    vec = np.asarray(vec, dtype=np.float32)
    east = vec[..., 0]
    up = vec[..., 1]
    north = -vec[..., 2]
    return np.stack((east, north, up), axis=-1)


def enu_to_visual_world(east: float, north: float, up: float) -> np.ndarray:
    return np.array([east, up, -north], dtype=np.float32)


def cubemap_face_rays(face_size: int = FACE_SIZE) -> np.ndarray:
    """(6, face_size, face_size, 3) unit rays in the aircraft visual frame."""
    bases = face_bases()
    # Pixel centers, u right, v up in the face.
    ids = (np.arange(face_size, dtype=np.float32) + 0.5) / face_size
    u = 2.0 * ids - 1.0
    v = 1.0 - 2.0 * ids
    uu, vv = np.meshgrid(u, v)
    ones = np.ones_like(uu)
    local = np.stack((uu, vv, ones), axis=-1)  # (H, W, 3) in face (right, up, forward)
    rays = np.einsum("fij,hwj->fhwi", bases, local)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return (rays / np.maximum(norms, 1e-12)).astype(np.float32)


def atlas_indices_for_rays(rays: np.ndarray, face_size: int = FACE_SIZE) -> np.ndarray:
    """Nearest cubemap texel for each ray. ``rays`` shape (..., 3), local visual frame."""
    rays = np.asarray(rays, dtype=np.float32)
    bases = face_bases()
    # Score faces by alignment with face-forward.
    forwards = bases[:, 2]
    dots = rays @ forwards.T
    face = np.argmax(dots, axis=-1)
    z = np.take_along_axis(dots, face[..., None], axis=-1)[..., 0]
    z = np.maximum(z, 1e-6)
    rights = bases[:, 0]
    ups = bases[:, 1]
    u = np.sum(rays * rights[face], axis=-1) / z
    v = np.sum(rays * ups[face], axis=-1) / z
    x = np.clip(((u + 1.0) * 0.5 * face_size).astype(np.int32), 0, face_size - 1)
    y = np.clip(((1.0 - v) * 0.5 * face_size).astype(np.int32), 0, face_size - 1)
    col = face % ATLAS_COLS
    row = face // ATLAS_COLS
    px = col * face_size + x
    py = row * face_size + y
    width = face_size * ATLAS_COLS
    return (py * width + px).astype(np.int32)


def cone_rays(rays: np.ndarray, sigma_deg: float = ACCEPTANCE_SIGMA_DEG) -> np.ndarray:
    """Center + six ring samples per ray. Returns (N, 7, 3)."""
    rays = np.asarray(rays, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = np.cross(rays, up)
    pole = np.linalg.norm(tangent, axis=-1) < 1e-5
    if np.any(pole):
        tangent[pole] = np.cross(rays[pole], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-8)
    vertical = np.cross(tangent, rays)
    theta = np.arange(6, dtype=np.float32) * (np.pi / 3.0)
    radius = np.deg2rad(sigma_deg) * np.sqrt(2.0)
    ring = rays[:, None, :] * np.cos(radius) + np.sin(radius) * (
        tangent[:, None, :] * np.cos(theta)[None, :, None]
        + vertical[:, None, :] * np.sin(theta)[None, :, None]
    )
    out = np.concatenate((rays[:, None, :], ring), axis=1)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def cone_weights() -> np.ndarray:
    return np.array([0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125], dtype=np.float32)


@dataclass(frozen=True)
class CameraCalibration:
    """Inspectable constants stored with every retinal stimulus."""

    version: str = "flypilot-cubemap-v1"
    face_size: int = FACE_SIZE
    face_order: tuple[str, ...] = FACE_ORDER
    horizontal_fov_deg: float = HORIZONTAL_FOV_DEG
    overlap_deg: float = OVERLAP_DEG
    elevation_limit_deg: float = ELEVATION_LIMIT_DEG
    acceptance_sigma_deg: float = ACCEPTANCE_SIGMA_DEG
    eye_up_m: float = EYE_UP_M
    eye_forward_m: float = EYE_FORWARD_M
    registration: str = (
        "Approximate angular registration: R1-R6 sorted by body_id within each "
        "annotated eye, stretched over the NeuroMechFly-inspired field. Not a "
        "measured MaleCNS optical-axis map (optic-column Excel is not loaded)."
    )
    color: str = "engineered RGB; no UV. Luminance is Rec.709 luma."

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "face_size": self.face_size,
            "face_order": list(self.face_order),
            "atlas_width": ATLAS_WIDTH,
            "atlas_height": ATLAS_HEIGHT,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "overlap_deg": self.overlap_deg,
            "elevation_limit_deg": self.elevation_limit_deg,
            "left_azimuth_deg": list(LEFT_AZIMUTH),
            "right_azimuth_deg": list(RIGHT_AZIMUTH),
            "acceptance_sigma_deg": self.acceptance_sigma_deg,
            "eye_up_m": self.eye_up_m,
            "eye_forward_m": self.eye_forward_m,
            "registration": self.registration,
            "color": self.color,
        }
