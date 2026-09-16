"""Record ExpertLandingController time-series for later imitation learning.

This dataset is expert *actions* from a conventional autopilot paired with
JSBSim aircraft state. It is NOT the dataset that trains the fly: a later
milestone will pair MaleCNS neural activity with these actions.

Format: Apache Parquet (columnar, efficient, easy to load in Python).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.initial_conditions import DECODER_SPAWN
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus

COLUMNS = [
    "episode_id",
    "seed",
    "sim_time_s",
    "along_m",
    "right_m",
    "east_m",
    "north_m",
    "up_m",
    "alt_agl_m",
    "alt_msl_m",
    "lat_deg",
    "lon_deg",
    "airspeed_kts",
    "groundspeed_kts",
    "vertical_speed_fpm",
    "pitch_deg",
    "roll_deg",
    "heading_deg",
    "alpha_deg",
    "beta_deg",
    "p_deg_s",
    "q_deg_s",
    "r_deg_s",
    "on_ground",
    "engine_rpm",
    "thrust_lbs",
    "aileron",
    "elevator",
    "rudder",
    "throttle",
    "roll_command_deg",
    "pitch_command_deg",
    "target_airspeed_kts",
    "throttle_trim",
    "phase",
    "outcome",
]


def _row(episode_id: int, seed: int, sandbox: LandingSandbox, outcome: str) -> dict:
    obs = sandbox.aircraft.observe()
    extra = obs.extra or {}
    phase = "n/a"
    tel = getattr(sandbox.controller, "telemetry", lambda: {})()
    if isinstance(tel, dict):
        phase = str(tel.get("phase", "n/a"))
    return {
        "episode_id": episode_id,
        "seed": seed,
        "sim_time_s": obs.sim_time_s,
        "along_m": obs.along_m,
        "right_m": obs.right_m,
        "east_m": obs.east_m,
        "north_m": obs.north_m,
        "up_m": obs.up_m,
        "alt_agl_m": obs.alt_agl_m,
        "alt_msl_m": obs.alt_msl_m,
        "lat_deg": obs.lat_deg,
        "lon_deg": obs.lon_deg,
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
        "engine_rpm": float(extra.get("engine_rpm", 0.0)),
        "thrust_lbs": float(extra.get("thrust_lbs", 0.0)),
        "aileron": obs.aileron,
        "elevator": obs.elevator,
        "rudder": obs.rudder,
        "throttle": obs.throttle,
        "roll_command_deg": float(tel.get("roll_command_deg", 0.0)),
        "pitch_command_deg": float(tel.get("pitch_command_deg", 0.0)),
        "target_airspeed_kts": float(tel.get("target_airspeed_kts", 0.0)),
        "throttle_trim": float(tel.get("throttle_trim", 0.0)),
        "phase": phase,
        "outcome": outcome,
    }


def record_episodes(
    output: Path,
    episodes: int,
    seed: int,
    success_only: bool = False,
    stride: int = 2,
    wide_spawns: bool = False,
) -> dict:
    """Run expert landings and write one Parquet file.

    `stride` keeps every Nth FDM step (default 2 → 60 Hz) so files stay small
    enough for later imitation learning without aliasing the approach.
    """
    sandbox = LandingSandbox(
        controller=ExpertLandingController(),
        randomize_spawns=True,
        spawn_spec=DECODER_SPAWN if wide_spawns else None,
    )
    chunks: list[pa.Table] = []
    kept = 0
    successes = 0
    for i in range(episodes):
        ep_seed = seed + i
        sandbox.reset(seed=ep_seed)
        frames: list[dict] = []
        step = 0
        while sandbox.episode.info.status is EpisodeStatus.IN_PROGRESS:
            sandbox.step_once()
            if step % stride == 0:
                frames.append(_row(i, ep_seed, sandbox, "in_progress"))
            step += 1
        outcome = sandbox.episode.info.status.value
        if sandbox.episode.info.status is EpisodeStatus.LANDED:
            successes += 1
        if success_only and outcome != EpisodeStatus.LANDED.value:
            continue
        for frame in frames:
            frame["outcome"] = outcome
        # Terminal frame always recorded.
        frames.append(_row(i, ep_seed, sandbox, outcome))
        chunks.append(pa.Table.from_pylist(frames, schema=_schema()))
        kept += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if chunks:
        table = pa.concat_tables(chunks)
        pq.write_table(table, output)
        rows = table.num_rows
    else:
        pq.write_table(pa.Table.from_pylist([], schema=_schema()), output)
        rows = 0
    return {
        "path": str(output),
        "episodes_requested": episodes,
        "episodes_written": kept,
        "successes": successes,
        "rows": rows,
        "stride": stride,
        "spawn_distribution": "decoder_wide" if wide_spawns else "default",
        "controller": "ExpertLandingController",
        "kind": "conventional_autopilot",
        "male_cns": False,
        "note": (
            "Expert actions from a classical autopilot. Not MaleCNS training data; "
            "later paired with connectome activity."
        ),
    }


def _schema() -> pa.Schema:
    fields = []
    for name in COLUMNS:
        if name in ("episode_id", "seed"):
            fields.append(pa.field(name, pa.int32()))
        elif name in ("on_ground",):
            fields.append(pa.field(name, pa.bool_()))
        elif name in ("phase", "outcome"):
            fields.append(pa.field(name, pa.string()))
        else:
            fields.append(pa.field(name, pa.float64()))
    return pa.schema(fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record ExpertLandingController demonstrations (classical autopilot, not MaleCNS)"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/expert/demonstrations.parquet"))
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--stride", type=int, default=2, help="Keep every Nth FDM step (2 = 60 Hz)")
    parser.add_argument("--wide-spawns", action="store_true", help="Use the wider held-out controller spawn distribution")
    args = parser.parse_args(argv)
    info = record_episodes(args.output, args.episodes, args.seed, args.success_only, args.stride, args.wide_spawns)
    print(json_dumps(info))
    return 0


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
