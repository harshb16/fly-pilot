"""Validation experiments A–D for fly visual observation.

These check the engineered visual path, not biological steering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.brain.vision.encoder import FlyEyeEncoder
from fly_pilot.brain.vision.geometry import LEFT_EYE_YAW_DEG, RIGHT_EYE_YAW_DEG
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.brain.vision.scene import FlyViewPose, render_cubemap_atlas, render_eye_preview
from fly_pilot.brain.vision.stimulus import RetinalStimulusFrame
from fly_pilot.record_observing import record_episodes, synthetic_observer
from fly_pilot.runway import Runway
from fly_pilot.brain.replay_episode import replay_table


def _approach_pose(*, extra_yaw: float = 0.0, east: float | None = None, heading: float = 90.0) -> FlyViewPose:
    # Default short-final: 3200 m west of a 090 runway, 250 m AGL.
    return FlyViewPose(
        east_m=-3200.0 if east is None else east,
        north_m=0.0,
        up_m=250.0,
        heading_deg=heading,
        pitch_deg=-4.5,
        roll_deg=0.0,
        extra_yaw_deg=extra_yaw,
    )


def experiment_a_static(observer: ObservingMaleCNS) -> dict:
    """Hold pose fixed: retinal stimulus and checksums must settle/reproduce."""
    observer.reset(seed=64)
    pose = _approach_pose()
    frames: list[RetinalStimulusFrame] = []
    checksums: list[str] = []
    for i in range(12):
        step = observer.step_pose(pose, sim_time_s=i * observer.config.dt, neural_step=i)
        frames.append(step.retinal)
        checksums.append(step.spike_checksum)
    lum = [float(f.luminance.mean()) for f in frames]
    currents = [f.current_sha256 for f in frames]
    # After the first frame, temporal contrast should drop toward a static scene.
    temporal = [float(observer.encoder.last_temporal.mean())]
    observer.encoder.reset()
    observer.reset(seed=64)
    checksums2 = [
        observer.step_pose(pose, sim_time_s=i * observer.config.dt, neural_step=i).spike_checksum
        for i in range(12)
    ]
    late_temporal = []
    observer.reset(seed=7)
    observer.encoder.reset()
    for i in range(8):
        observer.step_pose(pose, sim_time_s=i * 0.02)
        late_temporal.append(float(observer.encoder.last_temporal.mean()))
    return {
        "name": "A_static_scene",
        "luminance_stable": max(lum) - min(lum) < 1e-6,
        "current_stable_after_first": len(set(currents[1:])) == 1,
        "replay_same_seed_matches": checksums == checksums2,
        "temporal_drops": late_temporal[-1] <= late_temporal[0] + 1e-9,
        "mean_luminance": lum[-1],
        "n_steps": 12,
    }


def experiment_b_runway_motion(observer: ObservingMaleCNS) -> dict:
    """Translate toward the runway: stimulus must change."""
    observer.reset(seed=64)
    hashes = []
    lums = []
    spikes = []
    # Descend and close in so the runway/ground fill grows.
    for i, (east, up) in enumerate(zip(np.linspace(-3200.0, -80.0, 12), np.linspace(250.0, 20.0, 12))):
        pose = FlyViewPose(
            east_m=float(east),
            north_m=0.0,
            up_m=float(up),
            heading_deg=90.0,
            pitch_deg=-6.0,
            roll_deg=0.0,
        )
        step = observer.step_pose(pose, sim_time_s=i * 0.02)
        hashes.append(step.retinal.current_sha256)
        lums.append(float(step.retinal.luminance.mean()))
        spikes.append(step.n_spikes)
    return {
        "name": "B_runway_motion",
        "unique_retinal_frames": len(set(hashes)),
        "stimulus_changes": len(set(hashes)) > 1,
        "neural_changes": len(set(spikes)) > 1 or max(lums) - min(lums) > 1e-4,
        "luminance_series": lums,
        "n_spikes_series": spikes,
    }


def experiment_c_left_right(encoder: FlyEyeEncoder | None = None) -> dict:
    """Yaw so the runway sits left vs right of the nose. Spatial difference only."""
    mapping = encoder.mapping if encoder is not None else PhotoreceptorMap.synthetic(24, 24)
    enc = encoder or FlyEyeEncoder(mapping)
    runway = Runway()
    left_pose = _approach_pose(extra_yaw=-35.0)
    right_pose = _approach_pose(extra_yaw=35.0)
    left_atlas = render_cubemap_atlas(left_pose, runway)
    right_atlas = render_cubemap_atlas(right_pose, runway)
    enc.reset()
    left_frame = enc.encode(left_atlas)
    enc.reset()
    right_frame = enc.encode(right_atlas)
    left_mask = mapping.left_mask()
    right_mask = mapping.right_mask()
    # Difference of the two retinal images.
    delta = np.abs(left_frame.luminance - right_frame.luminance)
    spatial = bool(delta.mean() > 1e-4)
    # Left-eye vs right-eye mean under each bias — reported, not claimed as taxis.
    report = {
        "name": "C_left_right_perturbation",
        "atlas_differs": left_atlas.tobytes() != right_atlas.tobytes(),
        "retinal_spatial_difference": spatial,
        "mean_abs_luminance_delta": float(delta.mean()),
        "left_bias": {
            "left_eye_luminance": float(left_frame.luminance[left_mask].mean()),
            "right_eye_luminance": float(left_frame.luminance[right_mask].mean()),
        },
        "right_bias": {
            "left_eye_luminance": float(right_frame.luminance[left_mask].mean()),
            "right_eye_luminance": float(right_frame.luminance[right_mask].mean()),
        },
        "note": (
            "Spatial retinal difference is required. A particular biological "
            "steering reflex is NOT claimed."
        ),
    }
    # Previews exist so the UI can show left/right discs.
    _ = render_eye_preview(left_pose, runway, yaw_deg=LEFT_EYE_YAW_DEG)
    _ = render_eye_preview(right_pose, runway, yaw_deg=RIGHT_EYE_YAW_DEG)
    return report


def experiment_d_replay(tmp_parquet: Path) -> dict:
    meta = record_episodes(tmp_parquet, episodes=1, seed=3, synthetic=True, max_steps=240)
    replay = replay_table(tmp_parquet)
    return {
        "name": "D_replay",
        "record": {"rows": meta["rows"], "path": meta["path"]},
        "replay_ok": replay["ok"],
        "compared": replay["compared"],
        "matches": replay["matches"],
        "mismatches": replay["mismatches"][:3],
    }


def run_all(output: Path | None = None) -> dict:
    observer = synthetic_observer(seed=64)
    report = {
        "A": experiment_a_static(observer),
        "B": experiment_b_runway_motion(observer),
        "C": experiment_c_left_right(observer.encoder),
        "D": experiment_d_replay(Path("artifacts/vision-replay-smoke.parquet")),
    }
    report["all_passed"] = bool(
        report["A"]["replay_same_seed_matches"]
        and report["A"]["current_stable_after_first"]
        and report["B"]["stimulus_changes"]
        and report["C"]["retinal_spatial_difference"]
        and report["D"]["replay_ok"]
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/vision-experiments.json"))
    args = parser.parse_args(argv)
    report = run_all(args.json_out)
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
