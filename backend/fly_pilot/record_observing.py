"""Record EXPERT + FLY OBSERVING episodes.

ExpertLandingController flies JSBSim. MaleCNS observes a rendered fly-view.
The connectome never writes inceptors. Output is a new dataset format for a
later decoder — this command does not train one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fly_pilot.brain.config import LIFConfig
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus

SCHEMA_VERSION = "observing-expert-v1"


def _blob(array: np.ndarray, dtype) -> bytes:
    return np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes()


def _row(
    episode_id: int,
    seed: int,
    sandbox: LandingSandbox,
    outcome: str,
) -> dict[str, Any]:
    obs = sandbox.aircraft.observe()
    extra = obs.extra or {}
    phase = "n/a"
    tel = getattr(sandbox.controller, "telemetry", lambda: {})()
    if isinstance(tel, dict):
        phase = str(tel.get("phase", "n/a"))
    step = sandbox.observer.last_step if sandbox.observer is not None else None
    retinal = step.retinal if step is not None else None
    dn_vec = sandbox.observer.dn.feature_vector() if sandbox.observer is not None else np.zeros(0, np.float32)
    vis = step.visual_rates if step is not None else {}
    return {
        "episode_id": episode_id,
        "seed": seed,
        "sim_time_s": obs.sim_time_s,
        "neural_step": -1 if step is None else step.neural_step,
        "along_m": obs.along_m,
        "right_m": obs.right_m,
        "east_m": obs.east_m,
        "north_m": obs.north_m,
        "up_m": obs.up_m,
        "alt_agl_m": obs.alt_agl_m,
        "alt_msl_m": obs.alt_msl_m,
        "airspeed_kts": obs.airspeed_kts,
        "groundspeed_kts": obs.groundspeed_kts,
        "vertical_speed_fpm": obs.vertical_speed_fpm,
        "pitch_deg": obs.pitch_deg,
        "roll_deg": obs.roll_deg,
        "heading_deg": obs.heading_deg,
        "alpha_deg": obs.alpha_deg,
        "beta_deg": obs.beta_deg,
        "p_deg_s": obs.p_deg_s,
        "q_deg_s": obs.q_deg_s,
        "r_deg_s": obs.r_deg_s,
        "on_ground": obs.on_ground,
        "aileron": obs.aileron,
        "elevator": obs.elevator,
        "rudder": obs.rudder,
        "throttle": obs.throttle,
        "phase": phase,
        "outcome": outcome,
        "retinal_schema": "retinal-v1" if retinal is not None else "",
        "retinal_n": 0 if retinal is None else retinal.n_receptors,
        "retinal_current_sha256": "" if retinal is None else retinal.current_sha256,
        "atlas_sha256": "" if retinal is None else retinal.atlas_sha256,
        "retinal_currents_f16": b"" if retinal is None else _blob(retinal.currents, np.float16),
        "retinal_luminance_f16": b"" if retinal is None else _blob(retinal.luminance, np.float16),
        "dn_n": int(dn_vec.size),
        "dn_rates_f16": _blob(dn_vec, np.float16),
        "visual_rates_json": json.dumps(vis, sort_keys=True),
        "n_spikes": 0 if step is None else step.n_spikes,
        "n_outgoing_edges": 0 if step is None else step.n_outgoing_edges,
        "spike_checksum": "" if step is None else step.spike_checksum,
        "mean_luminance": 0.0 if retinal is None else float(retinal.luminance.mean()),
        "engine_rpm": float(extra.get("engine_rpm", 0.0)),
    }


def _schema() -> pa.Schema:
    fields = [
        pa.field("episode_id", pa.int32()),
        pa.field("seed", pa.int32()),
        pa.field("sim_time_s", pa.float64()),
        pa.field("neural_step", pa.int32()),
        pa.field("along_m", pa.float64()),
        pa.field("right_m", pa.float64()),
        pa.field("east_m", pa.float64()),
        pa.field("north_m", pa.float64()),
        pa.field("up_m", pa.float64()),
        pa.field("alt_agl_m", pa.float64()),
        pa.field("alt_msl_m", pa.float64()),
        pa.field("airspeed_kts", pa.float64()),
        pa.field("groundspeed_kts", pa.float64()),
        pa.field("vertical_speed_fpm", pa.float64()),
        pa.field("pitch_deg", pa.float64()),
        pa.field("roll_deg", pa.float64()),
        pa.field("heading_deg", pa.float64()),
        pa.field("alpha_deg", pa.float64()),
        pa.field("beta_deg", pa.float64()),
        pa.field("p_deg_s", pa.float64()),
        pa.field("q_deg_s", pa.float64()),
        pa.field("r_deg_s", pa.float64()),
        pa.field("on_ground", pa.bool_()),
        pa.field("aileron", pa.float64()),
        pa.field("elevator", pa.float64()),
        pa.field("rudder", pa.float64()),
        pa.field("throttle", pa.float64()),
        pa.field("phase", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("retinal_schema", pa.string()),
        pa.field("retinal_n", pa.int32()),
        pa.field("retinal_current_sha256", pa.string()),
        pa.field("atlas_sha256", pa.string()),
        pa.field("retinal_currents_f16", pa.binary()),
        pa.field("retinal_luminance_f16", pa.binary()),
        pa.field("dn_n", pa.int32()),
        pa.field("dn_rates_f16", pa.binary()),
        pa.field("visual_rates_json", pa.string()),
        pa.field("n_spikes", pa.int32()),
        pa.field("n_outgoing_edges", pa.int64()),
        pa.field("spike_checksum", pa.string()),
        pa.field("mean_luminance", pa.float64()),
        pa.field("engine_rpm", pa.float64()),
    ]
    return pa.schema(fields)


def synthetic_observer(n_left: int = 8, n_right: int = 8, n_extra: int = 24, seed: int = 64) -> ObservingMaleCNS:
    """Tiny graph for tests. Not MaleCNS."""
    from scipy import sparse

    n_r = n_left + n_right
    n = n_r + n_extra
    body_ids = np.arange(10, 10 + n, dtype=np.int64)
    types = np.array(["R1-R6"] * n_r + ["L1"] * (n_extra // 2) + ["DNx"] * (n_extra - n_extra // 2), dtype=object)
    superclasses = np.array(
        ["ol_sensory"] * n_r
        + ["ol_intrinsic"] * (n_extra // 2)
        + ["descending_neuron"] * (n_extra - n_extra // 2),
        dtype=object,
    )
    sides = np.array(["L"] * n_left + ["R"] * n_right + ["L"] * (n_extra // 2) + ["R"] * (n_extra - n_extra // 2), dtype=object)
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, 80, dtype=np.int32)
    post = rng.integers(0, n, 80, dtype=np.int32)
    w = rng.integers(1, 5, 80).astype(np.float32)
    counts = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    connectome = Connectome.from_parts(
        body_ids,
        counts,
        type=types,
        flywire_type=types,
        instance=np.array([f"{t}_{s}" for t, s in zip(types, sides)], dtype=object),
        superclass=superclasses,
        side=sides,
        consensus_nt=np.array(["histamine"] * n_r + ["acetylcholine"] * n_extra, dtype=object),
        sign=np.array([-1] * n_r + [1] * n_extra, dtype=np.int8),
    )
    mapping = PhotoreceptorMap.synthetic(n_left, n_right, start_index=0)
    mapping.indices = np.arange(n_r, dtype=np.int32)
    mapping.body_ids = body_ids[:n_r]
    return ObservingMaleCNS(
        connectome,
        config=LIFConfig(seed=seed, tonic_current=0.05, background_rate_hz=0.0),
        mapping=mapping,
    )


def record_episodes(
    output: Path,
    episodes: int,
    seed: int,
    *,
    success_only: bool = False,
    synthetic: bool = False,
    observer: ObservingMaleCNS | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if observer is None:
        observer = synthetic_observer(seed=seed) if synthetic else ObservingMaleCNS.load()
    sandbox = LandingSandbox(
        controller=ExpertLandingController(),
        randomize_spawns=True,
        observer=observer,
        observing=True,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[pa.Table] = []
    kept = 0
    successes = 0
    t0 = time.perf_counter()
    spike_sum = 0
    spike_max = 0
    neural_rows = 0
    for i in range(episodes):
        ep_seed = seed + i
        sandbox.reset(seed=ep_seed)
        frames: list[dict] = []
        last_neural = -1
        steps = 0
        while sandbox.episode.info.status is EpisodeStatus.IN_PROGRESS:
            sandbox.step_once()
            step = sandbox.observer.last_step
            if step is not None and step.neural_step != last_neural:
                frames.append(_row(i, ep_seed, sandbox, "in_progress"))
                last_neural = step.neural_step
                spike_sum += step.n_spikes
                spike_max = max(spike_max, step.n_spikes)
                neural_rows += 1
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        outcome = sandbox.episode.info.status.value
        if sandbox.episode.info.status is EpisodeStatus.LANDED:
            successes += 1
        if success_only and outcome != EpisodeStatus.LANDED.value:
            continue
        for frame in frames:
            frame["outcome"] = outcome
        frames.append(_row(i, ep_seed, sandbox, outcome))
        chunks.append(pa.Table.from_pylist(frames, schema=_schema()))
        kept += 1

    wall = time.perf_counter() - t0
    if chunks:
        table = pa.concat_tables(chunks)
        pq.write_table(table, output)
        rows = table.num_rows
        sim_time = float(table["sim_time_s"].to_numpy().max()) if rows else 0.0
    else:
        pq.write_table(pa.Table.from_pylist([], schema=_schema()), output)
        rows = 0
        sim_time = 0.0

    meta = {
        "schema": SCHEMA_VERSION,
        "path": str(output),
        "episodes_requested": episodes,
        "episodes_written": kept,
        "successes": successes,
        "rows": rows,
        "neural_rows": neural_rows,
        "wall_seconds": round(wall, 4),
        "approx_sim_seconds": round(sim_time, 4),
        "mean_spikes_per_neural_step": round(spike_sum / max(neural_rows, 1), 2),
        "max_spikes_per_neural_step": spike_max,
        "controller": "ExpertLandingController",
        "kind": "conventional_autopilot",
        "male_cns_controls_aircraft": False,
        "synthetic": synthetic,
        "synthetic_graph_seed": seed if synthetic else None,
        "observer": sandbox.observer.metadata() if sandbox.observer is not None else {},
        "photoreceptor_body_ids": (
            sandbox.observer.mapping.body_ids.tolist() if sandbox.observer is not None else []
        ),
        "photoreceptor_sides": (
            list(sandbox.observer.mapping.sides) if sandbox.observer is not None else []
        ),
        "dn_body_ids": sandbox.observer.dn.spec.body_ids.tolist() if sandbox.observer is not None else [],
        "dn_sides": list(sandbox.observer.dn.spec.sides) if sandbox.observer is not None else [],
        "dn_types": list(sandbox.observer.dn.spec.types) if sandbox.observer is not None else [],
        "dn_windows": list(sandbox.observer.dn.windows) if sandbox.observer is not None else [],
        "note": (
            "Fly-observing-expert dataset. Expert actions are classical autopilot. "
            "Retinal currents are the sensory input MaleCNS received. Replay with "
            "python -m fly_pilot.brain.replay_episode."
        ),
    }
    sidecar = output.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    meta["sidecar"] = str(sidecar)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/observing/expert_observing.parquet"))
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="tiny graph, not MaleCNS")
    parser.add_argument("--max-steps", type=int, default=None, help="cap FDM steps (tests)")
    args = parser.parse_args(argv)
    info = record_episodes(
        args.output,
        args.episodes,
        args.seed,
        success_only=args.success_only,
        synthetic=args.synthetic,
        max_steps=args.max_steps,
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
