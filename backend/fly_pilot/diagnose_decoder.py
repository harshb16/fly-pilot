"""Diagnose DN feature information, action imbalance, and visual causality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.brain.vision.scene import FlyViewPose
from fly_pilot.record_observing import synthetic_observer
from fly_pilot.train_decoder import load_compact_tables


def _sample_aligned(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = sum(int(x.shape[0]) for x, _ in pairs)
    sampled_x: list[np.ndarray] = []
    sampled_y: list[np.ndarray] = []
    for x, y in pairs:
        n = int(x.shape[0])
        if n == 0:
            continue
        take = n if total <= max_rows else max(1, int(round(max_rows * n / total)))
        index = np.linspace(0, n - 1, min(take, n), dtype=np.int64)
        sampled_x.append(x[index])
        sampled_y.append(y[index])
    return np.concatenate(sampled_x, axis=0), np.concatenate(sampled_y, axis=0)


def _sample_chunks(chunks: list[np.ndarray], max_rows: int) -> np.ndarray:
    pairs = [(chunk, np.zeros((chunk.shape[0], 1), dtype=np.uint8)) for chunk in chunks if chunk.size]
    sampled, _ = _sample_aligned(pairs, max_rows)
    return sampled


def _sample_rows(bundle: dict[str, Any], max_rows: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(episode["x"], episode["y"]) for episode in bundle["episodes"].values()]
    x, y = _sample_aligned(pairs, max_rows)
    return x.astype(np.float64), y.astype(np.float64)


def dataset_diagnostics(paths: Sequence[Path]) -> dict[str, Any]:
    bundle = load_compact_tables(paths)
    x, y = _sample_rows(bundle)
    feature_std = x.std(axis=0)
    feature_mean = x.mean(axis=0)
    action_mean = y.mean(axis=0)
    action_std = y.std(axis=0)
    centered_x = x - feature_mean
    centered_y = y - action_mean
    covariance = centered_x.T @ centered_y / max(x.shape[0] - 1, 1)
    denom = feature_std[:, None] * action_std[None, :]
    correlation = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 1e-12)
    names = ("aileron", "elevator", "rudder", "throttle")
    correlations = {}
    imbalance = {}
    median = np.median(y, axis=0)
    for i, name in enumerate(names):
        abs_corr = np.abs(correlation[:, i])
        correlations[name] = {
            "max_abs": float(abs_corr.max(initial=0.0)),
            "p99_abs": float(np.quantile(abs_corr, 0.99)),
            "features_above_0_10": int((abs_corr >= 0.10).sum()),
        }
        deviation = np.abs(y[:, i] - median[i])
        imbalance[name] = {
            "mean": float(action_mean[i]),
            "std": float(action_std[i]),
            "median": float(median[i]),
            "fraction_over_0_05_from_median": float((deviation > 0.05).mean()),
            "p95_abs_from_median": float(np.quantile(deviation, 0.95)),
        }
    phase_chunks: dict[str, list[np.ndarray]] = {"opening": [], "middle": [], "terminal": []}
    for episode in bundle["episodes"].values():
        features = episode["x"]
        n = int(features.shape[0])
        if n == 0:
            continue
        opening_end = max(1, int(round(n * 0.20)))
        terminal_start = min(n - 1, int(round(n * 0.80)))
        phase_chunks["opening"].append(features[:opening_end])
        phase_chunks["middle"].append(features[opening_end:terminal_start])
        phase_chunks["terminal"].append(features[terminal_start:])
    phase_features: dict[str, Any] = {}
    for phase, chunks in phase_chunks.items():
        nonempty = [chunk for chunk in chunks if chunk.size]
        if not nonempty:
            phase_features[phase] = {"sampled_rows": 0}
            continue
        values = _sample_chunks(nonempty, 20_000)
        std = values.std(axis=0)
        phase_features[phase] = {
            "definition": (
                "first 20% of each episode"
                if phase == "opening"
                else "middle 60% of each episode" if phase == "middle" else "final 20% of each episode"
            ),
            "sampled_rows": int(values.shape[0]),
            "feature_std_p50": float(np.quantile(std, 0.50)),
            "feature_std_p95": float(np.quantile(std, 0.95)),
            "silent_fraction": float((values <= 1e-6).mean()),
            "near_50hz_ceiling_fraction": float((values >= 49.9).mean()),
            "rate_p99_hz": float(np.quantile(values, 0.99)),
        }
    return {
        "datasets": bundle["datasets"],
        "episodes": bundle["n_episodes"],
        "rows": bundle["rows"],
        "sampled_rows": int(x.shape[0]),
        "dn_n": bundle["dn_n"],
        "features": {
            "count": int(x.shape[1]),
            "near_constant_fraction": float((feature_std < 1e-6).mean()),
            "zero_mean_fraction": float((np.abs(feature_mean) < 1e-8).mean()),
            "std_p50": float(np.quantile(feature_std, 0.50)),
            "std_p95": float(np.quantile(feature_std, 0.95)),
            "std_max": float(feature_std.max(initial=0.0)),
        },
        "feature_action_correlation": correlations,
        "action_imbalance": imbalance,
        "temporal_phase_features": phase_features,
    }


def offline_baselines(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    if "test" in payload:
        return payload["test"]
    offline = payload.get("offline")
    return offline.get("test") if isinstance(offline, dict) else None


def retina_to_dn_causality(observer: ObservingMaleCNS, *, steps: int = 24) -> dict[str, Any]:
    poses = {
        "left": FlyViewPose(-2200.0, 180.0, 180.0, 78.0, -4.0, 0.0),
        "right": FlyViewPose(-2200.0, -180.0, 180.0, 102.0, -4.0, 0.0),
    }
    vectors: dict[str, np.ndarray] = {}
    retinal_hashes: dict[str, str] = {}
    for name, pose in poses.items():
        observer.reset(seed=64)
        for step in range(steps):
            result = observer.step_pose(pose, sim_time_s=step * observer.config.dt, neural_step=step)
        vectors[name] = observer.dn.decoder_feature_vector().copy()
        retinal_hashes[name] = result.retinal.current_sha256
    delta = np.abs(vectors["left"] - vectors["right"])
    return {
        "poses": list(poses),
        "retinal_hashes_differ": retinal_hashes["left"] != retinal_hashes["right"],
        "dn_l2_delta": float(np.linalg.norm(delta)),
        "dn_changed_fraction": float((delta > 1e-6).mean()) if delta.size else 0.0,
        "dn_response_differs": bool(np.any(delta > 1e-6)),
    }


def termination_diagnostics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    rows = list(payload.get("episodes") or [])
    by_outcome: dict[str, list[float]] = {}
    for row in rows:
        by_outcome.setdefault(str(row.get("outcome")), []).append(float(row.get("elapsed_s") or 0.0))
    return {
        outcome: {
            "count": len(values),
            "median_elapsed_s": float(np.median(values)),
            "min_elapsed_s": float(np.min(values)),
        }
        for outcome, values in by_outcome.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", dest="data_paths")
    parser.add_argument("--fly-eval", type=Path, default=Path("artifacts/decoder/fly-eval.json"))
    parser.add_argument("--offline-eval", type=Path, default=Path("artifacts/decoder/offline_eval.json"))
    parser.add_argument("--synthetic-causality", action="store_true")
    parser.add_argument("--skip-causality", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("artifacts/decoder/diagnostics.json"))
    args = parser.parse_args(argv)
    data_paths = args.data_paths or [Path("data/decoder/expert_dn_controls.parquet")]
    report: dict[str, Any] = {
        "dataset": dataset_diagnostics(data_paths),
        "offline_baselines": offline_baselines(args.offline_eval),
        "termination": termination_diagnostics(args.fly_eval),
    }
    if not args.skip_causality:
        observer = synthetic_observer(seed=64) if args.synthetic_causality else ObservingMaleCNS.load()
        report["retina_to_dn"] = retina_to_dn_causality(observer)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
