"""Record compact DN-rate → expert-action pairs for decoder training.

ExpertLandingController flies JSBSim (with optional executed-command pulses).
MaleCNS observes the canonical fly-view. The compact table stores only
descending-neuron rates and expert inceptors — no retinal blobs, no aircraft
telemetry as decoder input.

A handful of full observing episodes can be written separately for replay.
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

from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.disturbance import ControlDisturbance, DisturbanceConfig
from fly_pilot.initial_conditions import DECODER_SPAWN, sample_spawn
from fly_pilot.record_observing import record_episodes as record_full_observing
from fly_pilot.record_observing import synthetic_observer
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus

SCHEMA_VERSION = "decoder-training-v1"
DECODER_INPUT_COLUMNS = ("dn_rates_100_f16", "dn_rates_260_f16")
DECODER_FORBIDDEN_INPUTS = (
    "along_m",
    "right_m",
    "alt_agl_m",
    "airspeed_kts",
    "pitch_deg",
    "roll_deg",
    "heading_deg",
    "p_deg_s",
    "q_deg_s",
    "r_deg_s",
)


def _blob(array: np.ndarray, dtype) -> bytes:
    return np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes()


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.int32()),
            pa.field("timestep", pa.int32()),
            pa.field("sim_time_s", pa.float64()),
            pa.field("dn_rates_100_f16", pa.binary()),
            pa.field("dn_rates_260_f16", pa.binary()),
            pa.field("expert_aileron", pa.float32()),
            pa.field("expert_elevator", pa.float32()),
            pa.field("expert_rudder", pa.float32()),
            pa.field("expert_throttle", pa.float32()),
        ]
    )


def _spawn_dict(spawn) -> dict[str, float | int | None]:
    return asdict(spawn)


def record_decoder_episodes(
    output: Path,
    successes: int,
    seed: int,
    *,
    max_attempts: int | None = None,
    synthetic: bool = False,
    observer=None,
    max_steps: int | None = None,
    disturb: bool = True,
    disturbance_config: DisturbanceConfig | None = None,
) -> dict[str, Any]:
    if observer is None:
        observer = synthetic_observer(seed=seed) if synthetic else None
        if observer is None:
            from fly_pilot.brain.observing import ObservingMaleCNS

            observer = ObservingMaleCNS.load()
    sandbox = LandingSandbox(
        controller=ExpertLandingController(),
        randomize_spawns=True,
        observer=observer,
        observing=True,
        spawn_spec=DECODER_SPAWN,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = max_attempts or max(successes * 3, successes + 8)
    chunks: list[pa.Table] = []
    episode_meta: list[dict[str, Any]] = []
    kept = 0
    attempts = 0
    t0 = time.perf_counter()
    attempt_seed = seed
    while kept < successes and attempts < max_attempts:
        ep_seed = attempt_seed
        attempt_seed += 1
        attempts += 1
        spawn = sample_spawn(sandbox.runway, sandbox.approach, seed=ep_seed, spec=DECODER_SPAWN)
        disturbance = ControlDisturbance.plan(
            ep_seed,
            config=disturbance_config,
            enabled=disturb and (ep_seed % 2 == 0),
        )
        sandbox.reset(seed=ep_seed)
        frames: list[dict[str, Any]] = []
        last_neural = -1
        steps = 0
        while sandbox.episode.info.status is EpisodeStatus.IN_PROGRESS:
            obs = sandbox.aircraft.observe()
            sandbox._run_neural_due(obs)
            sandbox.controller.observe(obs)
            expert = sandbox.controller.act()
            applied = disturbance.apply(expert, sim_time_s=obs.sim_time_s, alt_agl_m=obs.alt_agl_m)
            sandbox.last_applied_controls = applied
            sandbox.aircraft.apply_controls(applied)
            obs = sandbox.aircraft.step(1)
            sandbox.scheduler.note_physics(sandbox.aircraft.dt)
            sandbox.episode.update(obs)
            step = sandbox.observer.last_step if sandbox.observer is not None else None
            if step is not None and step.neural_step != last_neural:
                rates = sandbox.observer.dn.last_rates
                w_short, w_long = sandbox.observer.dn.windows[0], sandbox.observer.dn.windows[-1]
                short = rates.get(w_short, np.zeros(sandbox.observer.dn.spec.n, np.float32))
                long = rates.get(w_long, np.zeros(sandbox.observer.dn.spec.n, np.float32))
                frames.append(
                    {
                        "episode_id": kept,
                        "timestep": int(step.neural_step),
                        "sim_time_s": float(step.sim_time_s),
                        "dn_rates_100_f16": _blob(short, np.float16),
                        "dn_rates_260_f16": _blob(long, np.float16),
                        "expert_aileron": np.float32(expert.aileron),
                        "expert_elevator": np.float32(expert.elevator),
                        "expert_rudder": np.float32(expert.rudder),
                        "expert_throttle": np.float32(expert.throttle),
                    }
                )
                last_neural = step.neural_step
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        outcome = sandbox.episode.info.status.value
        info = sandbox.episode.info
        rec = {
            "attempt": attempts,
            "episode_id": kept if outcome == EpisodeStatus.LANDED.value or max_steps else None,
            "seed": ep_seed,
            "outcome": outcome,
            "reason": info.reason,
            "n_frames": len(frames),
            "spawn": _spawn_dict(spawn),
            "disturbances": disturbance.as_list(),
            "touchdown_fpm": info.touchdown_fpm,
        }
        if outcome == EpisodeStatus.LANDED.value or (max_steps is not None and frames):
            if frames:
                chunks.append(pa.Table.from_pylist(frames, schema=_schema()))
                rec["episode_id"] = kept
                rec["kept"] = True
                kept += 1
            else:
                rec["kept"] = False
        else:
            rec["kept"] = False
        episode_meta.append(rec)
        print(
            f"decoder-record attempt {attempts}/{max_attempts} seed={ep_seed} "
            f"{outcome} kept={kept}/{successes} frames={len(frames)}",
            flush=True,
        )

    wall = time.perf_counter() - t0
    if chunks:
        table = pa.concat_tables(chunks)
        pq.write_table(table, output)
        rows = table.num_rows
    else:
        pq.write_table(pa.Table.from_pylist([], schema=_schema()), output)
        rows = 0

    observer = sandbox.observer
    meta = {
        "schema": SCHEMA_VERSION,
        "path": str(output),
        "successes_requested": successes,
        "episodes_written": kept,
        "attempts": attempts,
        "rows": rows,
        "wall_seconds": round(wall, 4),
        "controller": "ExpertLandingController",
        "kind": "conventional_autopilot",
        "male_cns_controls_aircraft": False,
        "decoder_input": list(DECODER_INPUT_COLUMNS),
        "decoder_forbidden_inputs": list(DECODER_FORBIDDEN_INPUTS),
        "contains_retinal_blobs": False,
        "contains_aircraft_telemetry": False,
        "synthetic": synthetic,
        "spawn_spec": asdict(DECODER_SPAWN),
        "disturbance_config": asdict(disturbance_config or DisturbanceConfig()),
        "seed": seed,
        "episodes": episode_meta,
        "dn_body_ids": observer.dn.spec.body_ids.tolist() if observer is not None else [],
        "dn_sides": list(observer.dn.spec.sides) if observer is not None else [],
        "dn_types": list(observer.dn.spec.types) if observer is not None else [],
        "dn_windows": list(observer.dn.windows) if observer is not None else [],
        "dn_n": observer.dn.spec.n if observer is not None else 0,
        "observer": observer.metadata() if observer is not None else {},
        "note": (
            "Compact decoder-training set. Inputs are MaleCNS descending-neuron "
            "100 ms and 260 ms rates only. Targets are ExpertLandingController "
            "inceptors (classical autopilot). Not biological learning."
        ),
    }
    sidecar = output.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    meta["sidecar"] = str(sidecar)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/decoder/expert_dn_controls.parquet"))
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-disturb", action="store_true")
    parser.add_argument(
        "--full-replay-output",
        type=Path,
        default=None,
        help="Optional observing-v1 parquet with retinal blobs (few episodes).",
    )
    parser.add_argument("--full-replay-episodes", type=int, default=3)
    args = parser.parse_args(argv)
    info = record_decoder_episodes(
        args.output,
        args.successes,
        args.seed,
        max_attempts=args.max_attempts,
        synthetic=args.synthetic,
        max_steps=args.max_steps,
        disturb=not args.no_disturb,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "episodes"}, indent=2))
    if args.full_replay_output is not None:
        full = record_full_observing(
            args.full_replay_output,
            args.full_replay_episodes,
            args.seed + 50_000,
            success_only=True,
            synthetic=args.synthetic,
        )
        print(json.dumps({"full_replay": {k: full[k] for k in ("path", "episodes_written", "rows", "sidecar") if k in full}}, indent=2))
    return 0 if info["episodes_written"] >= min(args.successes, 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
