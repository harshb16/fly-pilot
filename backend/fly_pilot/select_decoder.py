"""Run the fixed four-candidate decoder matrix and select by closed-loop validation."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fly_pilot.evaluate_fly import _progress_score, evaluate_fly
from fly_pilot.train_decoder import PORTFOLIO_SEED_RANGES, train_decoder


@dataclass(frozen=True)
class Candidate:
    name: str
    data: tuple[Path, ...]
    hidden_linear: int
    gru_hidden: int


def _offline_mae(report: dict[str, Any]) -> float:
    controls = report["offline"]["test"]["gru"]
    return float(np.mean([float(metrics["mae"]) for metrics in controls.values()]))


def run_matrix(
    expert_data: Path,
    dagger_data: list[Path],
    output_dir: Path,
    *,
    epochs: int = 30,
    patience: int = 8,
    validation_seed: int = 3000,
    validation_episodes: int = 20,
    final_dir: Path | None = None,
) -> dict[str, Any]:
    if len(dagger_data) != 2:
        raise ValueError("the fixed portfolio experiment requires exactly two DAgger datasets")
    combined = (Path(expert_data), *[Path(path) for path in dagger_data])
    candidates = (
        Candidate("expert-full", (Path(expert_data),), 256, 128),
        Candidate("expert-small", (Path(expert_data),), 128, 64),
        Candidate("dagger-full", combined, 256, 128),
        Candidate("dagger-small", combined, 128, 64),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = output_dir / candidate.name
        checkpoint = candidate_dir / "best.pt"
        training_command = [
            "python",
            "-m",
            "fly_pilot.train_decoder",
            *[arg for path in candidate.data for arg in ("--data", str(path))],
            "--output",
            str(checkpoint),
            "--epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--hidden-linear",
            str(candidate.hidden_linear),
            "--gru-hidden",
            str(candidate.gru_hidden),
        ]
        report = train_decoder(
            candidate.data,
            checkpoint,
            epochs=epochs,
            patience=patience,
            hidden_linear=candidate.hidden_linear,
            gru_hidden=candidate.gru_hidden,
            training_command=training_command,
            seed_ranges=PORTFOLIO_SEED_RANGES,
        )
        rows, validation = evaluate_fly(
            validation_episodes,
            validation_seed,
            checkpoint=checkpoint,
            json_out=candidate_dir / "validation-eval.json",
            video_dir=candidate_dir / "videos",
            first_seed=validation_seed,
            include_first_attempt=False,
            write_videos=False,
        )
        progress = float(np.mean([_progress_score(row) for row in rows]))
        results.append(
            {
                "name": candidate.name,
                "checkpoint": str(checkpoint),
                "data": [str(path) for path in candidate.data],
                "hidden_linear": candidate.hidden_linear,
                "gru_hidden": candidate.gru_hidden,
                "success_count": int(validation["summary"]["success_count"]),
                "success_rate": float(validation["summary"]["success_rate"]),
                "mean_progress_score": progress,
                "offline_test_mean_mae": _offline_mae(report),
            }
        )
    ranked = sorted(
        results,
        key=lambda row: (
            -int(row["success_count"]),
            float(row["mean_progress_score"]),
            float(row["offline_test_mean_mae"]),
        ),
    )
    winner = ranked[0]
    winner_dir = Path(winner["checkpoint"]).parent
    final_dir = Path(final_dir or output_dir.parent)
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "best.meta.json", "train_report.json", "split.json"):
        shutil.copy2(winner_dir / name, final_dir / name)
    shutil.copy2(winner_dir / "validation-eval.json", final_dir / "validation-eval.json")
    report = {
        "selection_rule": [
            "highest 20-episode validation success count",
            "lowest mean progress score",
            "lowest offline test mean MAE",
        ],
        "validation_seed": validation_seed,
        "validation_episodes": validation_episodes,
        "candidates": results,
        "ranking": [row["name"] for row in ranked],
        "winner": winner,
    }
    (final_dir / "model-selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-data", type=Path, default=Path("data/decoder/expert_dn_controls.parquet"))
    parser.add_argument("--dagger-data", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/decoder/candidates"))
    parser.add_argument("--final-dir", type=Path, default=Path("artifacts/decoder"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.dagger_data:
        parser.error("pass both DAgger datasets with repeated --dagger-data")
    if len(args.dagger_data) != 2:
        parser.error("the fixed portfolio experiment requires exactly two DAgger datasets")
    report = run_matrix(
        args.expert_data,
        args.dagger_data,
        args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        final_dir=args.final_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
