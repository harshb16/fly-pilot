"""Collect on-policy DN-rate states with shadow-expert labels.

The trained MaleCNS controller is the sole JSBSim authority. A conventional
ExpertLandingController observes the same pre-action state and supplies only
the supervised target. Expert commands are never applied or blended.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fly_pilot.brain.decoder import DecoderArtifact, default_checkpoint_path, sha256_path
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.trained import TrainedMaleCNSController
from fly_pilot.initial_conditions import DECODER_SPAWN, sample_spawn
from fly_pilot.record_decoder import (
    DECODER_FORBIDDEN_INPUTS,
    DECODER_INPUT_COLUMNS,
    SCHEMA_VERSION,
    _blob,
    _schema,
)
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus


def record_dagger_episodes(
    output: Path,
    episodes: int,
    seed: int,
    *,
    checkpoint: Path | None = None,
    observer: ObservingMaleCNS | None = None,
    controller: TrainedMaleCNSController | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    observer = observer or ObservingMaleCNS.load()
    checkpoint = Path(checkpoint or default_checkpoint_path())
    controller = controller or TrainedMaleCNSController(observer, DecoderArtifact.load(checkpoint))
    if controller.name != "fly_control" or controller.uses_expert:
        raise RuntimeError("DAgger collection requires fly-only control authority")
    sandbox = LandingSandbox(
        controller=controller,
        observer=observer,
        observing=False,
        randomize_spawns=True,
        spawn_spec=DECODER_SPAWN,
        decoder_path=checkpoint,
    )
    shadow = ExpertLandingController(sandbox.runway)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, _schema(), compression="zstd")
    rows_written = 0
    episodes_written = 0
    episode_meta: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for episode_id in range(episodes):
        episode_seed = seed + episode_id
        spawn = sample_spawn(sandbox.runway, sandbox.approach, seed=episode_seed, spec=DECODER_SPAWN)
        sandbox.reset(seed=episode_seed)
        shadow.reset()
        frames: list[dict[str, Any]] = []
        last_neural = -1
        steps = 0
        while sandbox.episode.info.status is EpisodeStatus.IN_PROGRESS:
            obs = sandbox.aircraft.observe()
            sandbox._run_neural_due(obs)
            sandbox.controller.observe(obs)
            policy_controls = sandbox.controller.act()
            shadow.observe(obs)
            expert_target = shadow.act()
            step = observer.last_step
            if step is not None and step.neural_step != last_neural:
                rates = observer.dn.last_rates
                short_window, long_window = observer.dn.windows[0], observer.dn.windows[-1]
                short = rates.get(short_window, np.zeros(observer.dn.spec.n, np.float32))
                long = rates.get(long_window, np.zeros(observer.dn.spec.n, np.float32))
                frames.append(
                    {
                        "episode_id": episode_id,
                        "timestep": int(step.neural_step),
                        "sim_time_s": float(step.sim_time_s),
                        "dn_rates_100_f16": _blob(short, np.float16),
                        "dn_rates_260_f16": _blob(long, np.float16),
                        "expert_aileron": np.float32(expert_target.aileron),
                        "expert_elevator": np.float32(expert_target.elevator),
                        "expert_rudder": np.float32(expert_target.rudder),
                        "expert_throttle": np.float32(expert_target.throttle),
                    }
                )
                last_neural = int(step.neural_step)
            sandbox.last_applied_controls = policy_controls
            sandbox.aircraft.apply_controls(policy_controls)
            next_obs = sandbox.aircraft.step(1)
            sandbox.scheduler.note_physics(sandbox.aircraft.dt)
            sandbox.episode.update(next_obs)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if frames:
            episode_table = pa.Table.from_pylist(frames, schema=_schema())
            writer.write_table(episode_table)
            rows_written += episode_table.num_rows
            episodes_written += 1
        episode_meta.append(
            {
                "episode_id": episode_id,
                "seed": episode_seed,
                "outcome": sandbox.episode.info.status.value,
                "reason": sandbox.episode.info.reason,
                "n_frames": len(frames),
                "spawn": asdict(spawn),
                "policy_controls_aircraft": True,
                "expert_controls_aircraft": False,
            }
        )
        print(
            f"dagger-record {episode_id + 1}/{episodes} seed={episode_seed} "
            f"{sandbox.episode.info.status.value} frames={len(frames)}",
            flush=True,
        )

    writer.close()
    meta = {
        "schema": SCHEMA_VERSION,
        "collection": "dagger-shadow-expert-v1",
        "path": str(output),
        "sha256": sha256_path(output),
        "episodes_requested": episodes,
        "episodes_written": episodes_written,
        "rows": rows_written,
        "wall_seconds": round(time.perf_counter() - t0, 4),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint) if checkpoint.exists() else None,
        "controller": "TrainedMaleCNSController",
        "control_authority": "fly_control",
        "shadow_labeler": "ExpertLandingController",
        "expert_controls_aircraft": False,
        "commands_blended": False,
        "decoder_input": list(DECODER_INPUT_COLUMNS),
        "decoder_forbidden_inputs": list(DECODER_FORBIDDEN_INPUTS),
        "contains_aircraft_telemetry": False,
        "contains_retinal_blobs": False,
        "spawn_spec": asdict(DECODER_SPAWN),
        "dn_n": observer.dn.spec.n,
        "dn_windows": list(observer.dn.windows),
        "dn_body_ids": observer.dn.spec.body_ids.tolist(),
        "episodes": episode_meta,
        "note": (
            "The fly policy exclusively controlled JSBSim. The conventional expert "
            "only labeled visited states; its commands were never applied."
        ),
    }
    sidecar = output.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    meta["sidecar"] = str(sidecar)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1100)
    parser.add_argument("--output", type=Path, default=Path("data/decoder/dagger-round-1.parquet"))
    args = parser.parse_args(argv)
    report = record_dagger_episodes(
        args.output,
        args.episodes,
        args.seed,
        checkpoint=args.checkpoint,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    return 0 if report["episodes_written"] == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
