"""Headless evaluation of ExpertLandingController.

This measures whether the conventional autopilot can reliably land the
JSBSim C172 from modestly randomized short-final spawns. It is not a
MaleCNS evaluation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.guidance import heading_error_deg
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import EpisodeStatus


@dataclass
class EpisodeMetrics:
    episode_id: int
    seed: int
    outcome: str
    reason: str | None
    success: bool
    crash: bool
    failed_approach: bool
    centerline_error_m: float
    along_m: float
    heading_error_deg: float
    roll_deg: float
    pitch_deg: float
    touchdown_vertical_speed_fpm: float
    airspeed_kts: float
    elapsed_s: float
    alt_agl_m: float


def run_episode(sandbox: LandingSandbox, seed: int, episode_id: int) -> EpisodeMetrics:
    snap = sandbox.reset(seed=seed)
    runway_hdg = sandbox.runway.heading_deg
    while snap.episode.status is EpisodeStatus.IN_PROGRESS:
        snap = sandbox.step_once()
    obs = snap.observation
    status = snap.episode.status
    return EpisodeMetrics(
        episode_id=episode_id,
        seed=seed,
        outcome=status.value,
        reason=snap.episode.reason,
        success=status is EpisodeStatus.LANDED,
        crash=status is EpisodeStatus.CRASHED,
        failed_approach=status is EpisodeStatus.FAILED_APPROACH,
        centerline_error_m=obs.right_m,
        along_m=obs.along_m,
        heading_error_deg=heading_error_deg(obs.heading_deg, runway_hdg),
        roll_deg=obs.roll_deg,
        pitch_deg=obs.pitch_deg,
        touchdown_vertical_speed_fpm=obs.vertical_speed_fpm,
        airspeed_kts=obs.airspeed_kts,
        elapsed_s=obs.sim_time_s,
        alt_agl_m=obs.alt_agl_m,
    )


def summarize(rows: list[EpisodeMetrics]) -> dict:
    n = len(rows)
    successes = [r for r in rows if r.success]
    crashes = [r for r in rows if r.crash]
    failed = [r for r in rows if r.failed_approach]

    def _pct(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        k = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"mean": None, "p50": None, "p90": None, "p95": None, "worst": None}
        return {
            "mean": statistics.fmean(xs),
            "p50": _pct(xs, 50),
            "p90": _pct(xs, 90),
            "p95": _pct(xs, 95),
            "worst": max(xs, key=abs),
        }

    sink = [-r.touchdown_vertical_speed_fpm for r in successes]
    xtk = [abs(r.centerline_error_m) for r in successes]
    return {
        "episodes": n,
        "success_count": len(successes),
        "crash_count": len(crashes),
        "failed_approach_count": len(failed),
        "other_count": n - len(successes) - len(crashes) - len(failed),
        "success_rate": len(successes) / n if n else 0.0,
        "crash_rate": len(crashes) / n if n else 0.0,
        "failed_approach_rate": len(failed) / n if n else 0.0,
        "successful_touchdown": {
            "vertical_speed_fpm_sink": _stats(sink),
            "centerline_error_m_abs": _stats(xtk),
            "heading_error_deg_abs": _stats([abs(r.heading_error_deg) for r in successes]),
            "roll_deg_abs": _stats([abs(r.roll_deg) for r in successes]),
            "along_m": _stats([r.along_m for r in successes]),
            "airspeed_kts": _stats([r.airspeed_kts for r in successes]),
            "elapsed_s": _stats([r.elapsed_s for r in successes]),
        },
        "controller": {
            "name": "ExpertLandingController",
            "kind": "conventional_autopilot",
            "male_cns": False,
        },
    }


def format_report(summary: dict, rows: list[EpisodeMetrics]) -> str:
    td = summary["successful_touchdown"]
    ctl = summary.get("controller") or {}
    if ctl.get("name") == "TrainedMaleCNSController":
        heading = "FlyPilot FLY CONTROL evaluation"
        sub = "(fixed MaleCNS + trained temporal decoder — not biological learning; expert not in the loop)"
    else:
        heading = "FlyPilot ExpertLandingController evaluation"
        sub = "(conventional classical autopilot — not MaleCNS)"
    lines = [
        heading,
        sub,
        "",
        f"episodes:           {summary['episodes']}",
        f"success rate:       {100 * summary['success_rate']:.1f}%  ({summary['success_count']}/{summary['episodes']})",
        f"crash rate:         {100 * summary['crash_rate']:.1f}%  ({summary['crash_count']})",
        f"failed approach:    {100 * summary['failed_approach_rate']:.1f}%  ({summary['failed_approach_count']})",
    ]
    if td["vertical_speed_fpm_sink"]["mean"] is not None:
        vs = td["vertical_speed_fpm_sink"]
        xtk = td["centerline_error_m_abs"]
        lines += [
            "",
            "Successful touchdowns:",
            f"  mean sink:        {vs['mean']:.0f} fpm   (p90 {vs['p90']:.0f}, worst {vs['worst']:.0f})",
            f"  mean |xtk|:       {xtk['mean']:.2f} m    (p90 {xtk['p90']:.2f}, worst {xtk['worst']:.2f})",
            f"  mean |hdg err|:   {td['heading_error_deg_abs']['mean']:.2f} deg",
            f"  mean |roll|:      {td['roll_deg_abs']['mean']:.2f} deg",
            f"  mean along:       {td['along_m']['mean']:.0f} m",
            f"  mean IAS:         {td['airspeed_kts']['mean']:.1f} kt",
            f"  mean time:        {td['elapsed_s']['mean']:.1f} s",
        ]
    failures = [r for r in rows if not r.success]
    if failures:
        lines += ["", "Failures:"]
        for r in failures[:20]:
            lines.append(
                f"  ep {r.episode_id} seed={r.seed} {r.outcome}: {r.reason} "
                f"(along={r.along_m:.0f} xtk={r.centerline_error_m:.1f} vs={r.touchdown_vertical_speed_fpm:.0f})"
            )
        if len(failures) > 20:
            lines.append(f"  ... {len(failures) - 20} more")
    return "\n".join(lines) + "\n"


def evaluate(episodes: int, seed: int) -> tuple[list[EpisodeMetrics], dict]:
    sandbox = LandingSandbox(
        controller=ExpertLandingController(),
        randomize_spawns=True,
    )
    rows = [run_episode(sandbox, seed=seed + i, episode_id=i) for i in range(episodes)]
    return rows, summarize(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless ExpertLandingController evaluation (classical autopilot, not MaleCNS)"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Write full metrics JSON")
    args = parser.parse_args(argv)
    rows, summary = evaluate(args.episodes, args.seed)
    report = format_report(summary, rows)
    print(report, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "episodes": [asdict(r) for r in rows]}
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")
    return 0 if summary["success_rate"] >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
