"""Closed-loop evaluation of TrainedMaleCNSController.

Decoder outputs go to JSBSim. ExpertLandingController is not consulted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fly_pilot.brain.decoder import DecoderArtifact, default_checkpoint_path
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.controllers.trained import TrainedMaleCNSController
from fly_pilot.evaluate import EpisodeMetrics, format_report, summarize
from fly_pilot.guidance import heading_error_deg
from fly_pilot.initial_conditions import DECODER_SPAWN
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus
from fly_pilot.video import TraceFrame, write_approach_video


@dataclass
class FlyEpisodeResult:
    metrics: EpisodeMetrics
    frames: list[TraceFrame]


def run_fly_episode(
    sandbox: LandingSandbox,
    seed: int,
    episode_id: int,
    *,
    record: bool = False,
    sample_dt: float = 0.10,
) -> FlyEpisodeResult:
    if sandbox.controller.name != "fly_control":
        raise RuntimeError("FLY CONTROL evaluation must not use ExpertLandingController")
    from fly_pilot.controllers.expert import ExpertLandingController

    if isinstance(sandbox.controller, ExpertLandingController):
        raise RuntimeError("FLY CONTROL evaluation must not use ExpertLandingController")
    snap = sandbox.reset(seed=seed)
    frames: list[TraceFrame] = []
    last_t = -1e9
    while snap.episode.status is EpisodeStatus.IN_PROGRESS:
        snap = sandbox.step_once()
        obs = snap.observation
        if record and (obs.sim_time_s - last_t) >= sample_dt:
            applied = snap.applied_controls
            frames.append(
                TraceFrame(
                    sim_time_s=obs.sim_time_s,
                    along_m=obs.along_m,
                    right_m=obs.right_m,
                    alt_agl_m=obs.alt_agl_m,
                    heading_deg=obs.heading_deg,
                    airspeed_kts=obs.airspeed_kts,
                    aileron=0.0 if applied is None else applied.aileron,
                    elevator=0.0 if applied is None else applied.elevator,
                    rudder=0.0 if applied is None else applied.rudder,
                    throttle=0.0 if applied is None else applied.throttle,
                    status=snap.episode.status.value,
                )
            )
            last_t = obs.sim_time_s
    obs = snap.observation
    status = snap.episode.status
    metrics = EpisodeMetrics(
        episode_id=episode_id,
        seed=seed,
        outcome=status.value,
        reason=snap.episode.reason,
        success=status is EpisodeStatus.LANDED,
        crash=status is EpisodeStatus.CRASHED,
        failed_approach=status is EpisodeStatus.FAILED_APPROACH,
        centerline_error_m=obs.right_m,
        along_m=obs.along_m,
        heading_error_deg=heading_error_deg(obs.heading_deg, sandbox.runway.heading_deg),
        roll_deg=obs.roll_deg,
        pitch_deg=obs.pitch_deg,
        touchdown_vertical_speed_fpm=obs.vertical_speed_fpm,
        airspeed_kts=obs.airspeed_kts,
        elapsed_s=obs.sim_time_s,
        alt_agl_m=obs.alt_agl_m,
    )
    return FlyEpisodeResult(metrics=metrics, frames=frames)


def make_fly_sandbox(checkpoint: Path | None = None, observer: ObservingMaleCNS | None = None) -> LandingSandbox:
    observer = observer or ObservingMaleCNS.load()
    artifact = DecoderArtifact.load(checkpoint or default_checkpoint_path())
    controller = TrainedMaleCNSController(observer, artifact)
    return LandingSandbox(
        controller=controller,
        observer=observer,
        observing=False,
        randomize_spawns=True,
        spawn_spec=DECODER_SPAWN,
        decoder_path=checkpoint,
    )


def evaluate_fly(
    episodes: int,
    seed: int,
    *,
    checkpoint: Path | None = None,
    json_out: Path | None = None,
    video_dir: Path | None = None,
    observer: ObservingMaleCNS | None = None,
    first_seed: int = 1500,
    include_first_attempt: bool = True,
    write_videos: bool = True,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    sandbox = make_fly_sandbox(checkpoint, observer)
    if sandbox.controller.name != "fly_control":
        raise RuntimeError("expected fly_control authority")
    video_dir = Path(video_dir or Path("artifacts/decoder/videos"))
    if write_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    first: FlyEpisodeResult | None = None
    first_video: Path | None = None
    if include_first_attempt:
        print(f"fly-eval first attempt seed={first_seed} (held-out, training distribution)", flush=True)
        first = run_fly_episode(
            sandbox,
            seed=first_seed,
            episode_id=-1,
            record=write_videos,
        )
        if write_videos and first.frames:
            first_video = video_dir / "first_autonomous_attempt.gif"
            write_approach_video(first.frames, first_video, title="First FLY CONTROL attempt")
        print(
            f"fly-eval first {first.metrics.outcome} xtk={first.metrics.centerline_error_m:.1f}m "
            f"hdg={first.metrics.heading_error_deg:.1f}deg",
            flush=True,
        )

    rows: list[EpisodeMetrics] = []
    best: tuple[float, FlyEpisodeResult] | None = None
    closest: tuple[float, FlyEpisodeResult] | None = None
    for i in range(episodes):
        result = run_fly_episode(sandbox, seed=seed + i, episode_id=i, record=write_videos)
        rows.append(result.metrics)
        print(
            f"fly-eval {i+1}/{episodes} seed={seed+i} {result.metrics.outcome} "
            f"xtk={result.metrics.centerline_error_m:.1f}m",
            flush=True,
        )
        if result.metrics.success:
            score = _landing_score(result.metrics)
            if best is None or score < best[0]:
                best = (score, result)
        prog = _progress_score(result.metrics)
        if closest is None or prog < closest[0]:
            closest = (prog, result)

    summary = summarize(rows)
    summary["controller"] = {
        "name": "TrainedMaleCNSController",
        "kind": "fixed_malecns_plus_trained_temporal_decoder",
        "male_cns": True,
        "biological_learning": False,
        "expert_in_loop": False,
    }
    showcase = best[1] if best is not None else (closest[1] if closest is not None else None)
    best_video = None
    if write_videos and showcase is not None and showcase.frames:
        name = "best_successful_attempt.gif" if best is not None else "closest_attempt.gif"
        best_video = video_dir / name
        title = "Best FLY CONTROL landing" if best is not None else "Closest FLY CONTROL attempt"
        write_approach_video(showcase.frames, best_video, title=title)
    payload = {
        "first_attempt": None if first is None else asdict(first.metrics),
        "summary": summary,
        "episodes": [asdict(r) for r in rows],
        "videos": {
            "first": str(first_video) if first_video else None,
            "best": str(best_video) if best_video else None,
        },
        "best": None if best is None else asdict(best[1].metrics),
        "closest": None if closest is None else asdict(closest[1].metrics),
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return rows, payload


def _landing_score(row: EpisodeMetrics) -> float:
    return abs(row.centerline_error_m) + 0.05 * abs(row.heading_error_deg) + 0.002 * abs(row.touchdown_vertical_speed_fpm)


def _progress_score(row: EpisodeMetrics) -> float:
    """Lower is closer to a landing when no episode succeeded."""
    return (
        abs(row.centerline_error_m)
        + 0.4 * abs(row.heading_error_deg)
        - 0.02 * row.along_m
        + (0.0 if row.success else 500.0)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=Path("artifacts/decoder/fly-eval.json"))
    parser.add_argument("--video-dir", type=Path, default=Path("artifacts/decoder/videos"))
    args = parser.parse_args(argv)
    rows, payload = evaluate_fly(
        args.episodes,
        args.seed,
        checkpoint=args.checkpoint,
        json_out=args.json,
        video_dir=args.video_dir,
    )
    print(format_report(payload["summary"], rows), end="")
    print(f"videos: {payload['videos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
