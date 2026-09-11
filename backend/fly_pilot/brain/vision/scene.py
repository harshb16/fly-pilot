"""Software fly-view renderer of the landing scene.

REAL: runway rectangle, ground plane, sky, aircraft pose from JSBSim.
MODELED: Lambertian-ish unlit colors, no hills, no Cessna mesh (the fly is
looking *from* the airplane, so it should not see its own fuselage).

This is the canonical sensory renderer for MaleCNS. Three.js fly cameras are a
separate human visualization of the same geometry; they do not drive neural
time. Headless recording and replay use this module so a browser is not
required to reproduce retinal stimuli.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_pilot.brain.vision.geometry import (
    ATLAS_HEIGHT,
    ATLAS_WIDTH,
    EYE_FORWARD_M,
    EYE_UP_M,
    FACE_SIZE,
    PREVIEW_FOV_DEG,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    aircraft_rotation_visual,
    angular_rays,
    cubemap_face_rays,
    enu_to_visual_world,
    visual_world_to_enu,
)
from fly_pilot.runway import Runway
from fly_pilot.state import AircraftObservation

# Match Three.js ``scene.background`` / ground / pavement (docs/vision.md).
SKY_ZENITH = np.array([167, 196, 226], dtype=np.float32)
SKY_HORIZON = np.array([210, 224, 232], dtype=np.float32)
GROUND = np.array([138, 161, 109], dtype=np.float32)
GROUND_FAR = np.array([170, 186, 168], dtype=np.float32)
RUNWAY = np.array([48, 50, 58], dtype=np.float32)
SHOULDER = np.array([92, 95, 88], dtype=np.float32)
MARKING = np.array([245, 245, 240], dtype=np.float32)
# Directional sky bias (engineering sun) so yawing the eye changes the image.
SUN_ENU = np.array([0.55, 0.15, 0.82], dtype=np.float32)  # east, slightly north, up
SUN_ENU = SUN_ENU / np.linalg.norm(SUN_ENU)


@dataclass(frozen=True)
class FlyViewPose:
    """Aircraft-relative fly-eye origin in runway ENU."""

    east_m: float
    north_m: float
    up_m: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    extra_yaw_deg: float = 0.0

    @classmethod
    def from_observation(
        cls,
        obs: AircraftObservation,
        *,
        extra_yaw_deg: float = 0.0,
    ) -> "FlyViewPose":
        return cls(
            east_m=obs.east_m,
            north_m=obs.north_m,
            up_m=obs.up_m,
            heading_deg=obs.heading_deg,
            pitch_deg=obs.pitch_deg,
            roll_deg=obs.roll_deg,
            extra_yaw_deg=extra_yaw_deg,
        )


def _eye_origin_enu(pose: FlyViewPose) -> np.ndarray:
    rot = aircraft_rotation_visual(
        pose.heading_deg + pose.extra_yaw_deg,
        pose.pitch_deg,
        pose.roll_deg,
    )
    local_offset = np.array([0.0, EYE_UP_M, -EYE_FORWARD_M], dtype=np.float32)
    world = rot @ local_offset
    enu_off = visual_world_to_enu(world)
    return np.array(
        [pose.east_m + enu_off[0], pose.north_m + enu_off[1], pose.up_m + enu_off[2]],
        dtype=np.float32,
    )


def _local_rays_to_enu(pose: FlyViewPose, rays_local: np.ndarray) -> np.ndarray:
    rot = aircraft_rotation_visual(
        pose.heading_deg + pose.extra_yaw_deg,
        pose.pitch_deg,
        pose.roll_deg,
    )
    world = rays_local @ rot.T
    return visual_world_to_enu(world).astype(np.float32)


def shade_rays(
    origin_enu: np.ndarray,
    dirs_enu: np.ndarray,
    runway: Runway,
) -> np.ndarray:
    """Vectorized ground/runway/sky shading. Returns uint8 RGB (..., 3)."""
    dirs = np.asarray(dirs_enu, dtype=np.float32)
    orig = np.asarray(origin_enu, dtype=np.float32)
    east = dirs[..., 0]
    north = dirs[..., 1]
    up = dirs[..., 2]
    # Ground plane up=0. Hit if going downward from above.
    denom = up.copy()
    t = np.full(up.shape, np.inf, dtype=np.float32)
    going_down = denom < -1e-5
    t[going_down] = -orig[2] / denom[going_down]
    hit = going_down & (t > 0.05)

    hit_e = orig[0] + t * east
    hit_n = orig[1] + t * north
    heading = np.deg2rad(runway.heading_deg)
    s = np.sin(heading)
    c = np.cos(heading)
    along = np.zeros(up.shape, dtype=np.float32)
    right = np.zeros(up.shape, dtype=np.float32)
    along[hit] = hit_e[hit] * s + hit_n[hit] * c
    right[hit] = hit_e[hit] * c - hit_n[hit] * s

    rgb = np.zeros(dirs.shape, dtype=np.float32)
    elev = np.clip(up, 0.0, 1.0)
    sky = SKY_HORIZON * (1.0 - elev[..., None]) + SKY_ZENITH * elev[..., None]
    sun = np.clip(np.sum(dirs * SUN_ENU, axis=-1), 0.0, 1.0) ** 4
    sky = sky + (np.array([70.0, 55.0, 20.0], dtype=np.float32) * sun[..., None])
    rgb[:] = sky

    ground = np.broadcast_to(GROUND, rgb.shape).copy()
    if np.any(hit):
        dist = np.sqrt((hit_e[hit] - orig[0]) ** 2 + (hit_n[hit] - orig[1]) ** 2)
        fog = np.clip(dist / 6000.0, 0.0, 1.0).astype(np.float32)
        # World-space checker so translation and yaw change sampled luma even
        # when the runway patch is a few pixels (optic-flow cue).
        cell = (
            np.floor(along[hit] / 40.0).astype(np.int32)
            + np.floor(right[hit] / 40.0).astype(np.int32)
        ) & 1
        brighter = GROUND + np.array([22.0, 18.0, 12.0], dtype=np.float32)
        base = np.where(cell[:, None] == 0, GROUND, brighter)
        ground[hit] = base * (1.0 - fog[:, None]) + GROUND_FAR * fog[:, None]
    # Shoulder around runway.
    on_shoulder = (
        hit
        & (along >= -20.0)
        & (along <= runway.length_m + 20.0)
        & (np.abs(right) <= runway.width_m / 2.0 + 11.0)
    )
    ground[on_shoulder] = SHOULDER
    on_runway = (
        hit
        & (along >= 0.0)
        & (along <= runway.length_m)
        & (np.abs(right) <= runway.width_m / 2.0)
    )
    ground[on_runway] = RUNWAY
    # Centerline dashes every 40 m.
    dash = on_runway & (np.abs(right) < 0.45) & ((along % 40.0) < 18.0) & (along > 25.0)
    ground[dash] = MARKING
    # Threshold bars.
    thresh = on_runway & (along >= 18.0) & (along <= 40.0) & ((np.abs(right) % 2.6) < 0.9)
    ground[thresh] = MARKING

    rgb[hit] = ground[hit]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_cubemap_atlas(
    pose: FlyViewPose,
    runway: Runway | None = None,
    *,
    face_size: int = FACE_SIZE,
) -> np.ndarray:
    """Return (ATLAS_HEIGHT, ATLAS_WIDTH, 3) uint8 cubemap atlas."""
    runway = runway or Runway()
    rays_local = cubemap_face_rays(face_size)  # (6, H, W, 3)
    origin = _eye_origin_enu(pose)
    dirs_enu = _local_rays_to_enu(pose, rays_local.reshape(-1, 3)).reshape(rays_local.shape)
    faces = shade_rays(origin, dirs_enu, runway)
    atlas = np.zeros((ATLAS_HEIGHT, ATLAS_WIDTH, 3), dtype=np.uint8)
    for i in range(6):
        col = i % 3
        row = i // 3
        y0 = row * face_size
        x0 = col * face_size
        atlas[y0 : y0 + face_size, x0 : x0 + face_size] = faces[i]
    return atlas


def render_eye_preview(
    pose: FlyViewPose,
    runway: Runway | None = None,
    *,
    yaw_deg: float,
    width: int = PREVIEW_WIDTH,
    height: int = PREVIEW_HEIGHT,
    fov_deg: float = PREVIEW_FOV_DEG,
) -> np.ndarray:
    """Wide-FOV pinhole preview for one eye, uint8 (H, W, 3)."""
    runway = runway or Runway()
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    u = 2.0 * xs - 1.0
    v = 1.0 - 2.0 * ys
    uu, vv = np.meshgrid(u, v)
    # Horizontal FOV maps u=±1 to ±fov/2 at the image center.
    half = np.deg2rad(fov_deg / 2.0)
    aspect = width / max(height, 1)
    x = np.tan(half) * uu * aspect
    y = np.tan(half) * vv
    z = -np.ones_like(x)
    local = np.stack((x, y, z), axis=-1)
    local /= np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1e-8)
    # Yaw the eye around visual up.
    a = np.deg2rad(yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot_y = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)
    local = local @ rot_y.T
    origin = _eye_origin_enu(pose)
    dirs = _local_rays_to_enu(pose, local.reshape(-1, 3)).reshape(local.shape)
    return shade_rays(origin, dirs, runway)


def atlas_checksum(atlas: np.ndarray) -> str:
    import hashlib

    payload = np.ascontiguousarray(atlas.astype(np.uint8, copy=False))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def empty_atlas() -> np.ndarray:
    return np.zeros((ATLAS_HEIGHT, ATLAS_WIDTH, 3), dtype=np.uint8)
