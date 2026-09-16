"""Closed-loop evaluation for the task-trained MaleCNS population-graph policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fly_pilot.brain.graph_policy import default_graph_checkpoint_path
from fly_pilot.controllers.graph import ConnectomeGraphController
from fly_pilot.evaluate import EpisodeMetrics, format_report, run_episode, summarize
from fly_pilot.initial_conditions import DECODER_SPAWN
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus
from fly_pilot.video import TraceFrame, write_approach_video


def evaluate_graph(
    episodes: int,
    seed: int,
    *,
    checkpoint: Path | None = None,
    json_out: Path | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    path = checkpoint or default_graph_checkpoint_path()
    controller = ConnectomeGraphController.load(path)
    sandbox = LandingSandbox(
        controller=controller,
        randomize_spawns=True,
        spawn_spec=DECODER_SPAWN,
    )
    rows: list[EpisodeMetrics] = []
    for i in range(episodes):
        row = run_episode(sandbox, seed=seed + i, episode_id=i)
        rows.append(row)
        print(
            f"graph-eval {i + 1}/{episodes} seed={seed + i} {row.outcome} "
            f"xtk={row.centerline_error_m:.1f}m",
            flush=True,
        )
    summary = summarize(rows)
    summary["controller"] = {
        "name": "ConnectomeGraphController",
        "kind": controller.kind,
        "male_cns_topology": True,
        "fixed_malecns": False,
        "biological_learning": False,
        "expert_in_loop": False,
        "uses_aircraft_telemetry": True,
    }
    payload = {"summary": summary, "episodes": [asdict(row) for row in rows]}
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return rows, payload


def write_demo(checkpoint: Path, seed: int, output: Path) -> str:
    """Render one deterministic hybrid landing for the portfolio README."""
    controller = ConnectomeGraphController.load(checkpoint)
    sandbox = LandingSandbox(controller=controller, randomize_spawns=True, spawn_spec=DECODER_SPAWN)
    snap = sandbox.reset(seed=seed)
    frames: list[TraceFrame] = []
    last_time = -1e9
    while snap.episode.status is EpisodeStatus.IN_PROGRESS:
        snap = sandbox.step_once()
        obs = snap.observation
        if obs.sim_time_s - last_time >= 0.10:
            controls = snap.applied_controls
            frames.append(
                TraceFrame(
                    sim_time_s=obs.sim_time_s,
                    along_m=obs.along_m,
                    right_m=obs.right_m,
                    alt_agl_m=obs.alt_agl_m,
                    heading_deg=obs.heading_deg,
                    airspeed_kts=obs.airspeed_kts,
                    aileron=0.0 if controls is None else controls.aileron,
                    elevator=0.0 if controls is None else controls.elevator,
                    rudder=0.0 if controls is None else controls.rudder,
                    throttle=0.0 if controls is None else controls.throttle,
                    status=snap.episode.status.value,
                )
            )
            last_time = obs.sim_time_s
    output.parent.mkdir(parents=True, exist_ok=True)
    write_approach_video(frames, output, title="HYBRID FLY GUIDANCE landing")
    return snap.episode.status.value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--checkpoint", type=Path, default=default_graph_checkpoint_path())
    parser.add_argument("--json", type=Path, default=Path("artifacts/connectome_graph/validation.json"))
    parser.add_argument("--demo-gif", type=Path, default=None)
    parser.add_argument("--demo-seed", type=int, default=7000)
    args = parser.parse_args(argv)
    rows, payload = evaluate_graph(args.episodes, args.seed, checkpoint=args.checkpoint, json_out=args.json)
    print(format_report(payload["summary"], rows), end="")
    if args.demo_gif is not None:
        outcome = write_demo(args.checkpoint, args.demo_seed, args.demo_gif)
        print(f"demo: {args.demo_gif} ({outcome})")
    return 0 if payload["summary"]["success_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
