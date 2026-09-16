"""Offline decoder evaluation on held-out episodes.

Reports MAE / RMSE / R² / Pearson per control against:
1. the trained GRU (causal, full-episode)
2. a constant-mean action baseline
3. Ridge regression on the same DN features (benchmark only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from fly_pilot.brain.decoder import CONTROL_NAMES, CausalTemporalDecoder, DecoderArtifact, DecoderConfig
from fly_pilot.train_decoder import load_compact_tables, split_episode_ids

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CONTROL_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = float(np.sum((t - t.mean()) ** 2))
        r2 = float("nan") if denom < 1e-12 else float(1.0 - np.sum(err**2) / denom)
        t64 = t.astype(np.float64, copy=False)
        p64 = p.astype(np.float64, copy=False)
        tc = t64 - t64.mean()
        pc = p64 - p64.mean()
        corr_denom = float(np.sqrt(np.dot(tc, tc) * np.dot(pc, pc)))
        if t.size > 2 and corr_denom > 1e-12:
            pearson = float(np.dot(tc, pc) / corr_denom)
        else:
            pearson = float("nan")
        out[name] = {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}
    return out


def predict_gru_episode(model: CausalTemporalDecoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    model.reset_state()
    preds = []
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        y, _ = model(xt)
        preds = y.squeeze(0).cpu().numpy()
    model.reset_state()
    return np.asarray(preds, dtype=np.float32)


def ridge_fit(x: np.ndarray, y: np.ndarray, l2: float = 10.0) -> np.ndarray:
    n = x.shape[0]
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
    xtx = xb.T @ xb
    xtx[np.diag_indices_from(xtx)] += l2
    xty = xb.T @ y
    return np.linalg.solve(xtx, xty)


def ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
    return xb @ w


def stack_split(
    episodes: dict[int, dict[str, np.ndarray]],
    ids: list[int],
    *,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [episodes[i]["x"] for i in ids if i in episodes and episodes[i]["x"].size]
    ys = [episodes[i]["y"] for i in ids if i in episodes and episodes[i]["y"].size]
    if not xs:
        return np.zeros((0, 1), np.float32), np.zeros((0, 4), np.float32)
    total = sum(int(x.shape[0]) for x in xs)
    if max_rows is not None and total > max_rows:
        sampled_x: list[np.ndarray] = []
        sampled_y: list[np.ndarray] = []
        for x, y in zip(xs, ys, strict=True):
            take = max(1, int(round(max_rows * x.shape[0] / total)))
            index = np.linspace(0, x.shape[0] - 1, min(take, x.shape[0]), dtype=np.int64)
            sampled_x.append(x[index])
            sampled_y.append(y[index])
        xs, ys = sampled_x, sampled_y
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_splits(
    model: CausalTemporalDecoder,
    bundle: dict[str, Any],
    split: dict[str, list[int]],
    *,
    plots_dir: Path | None = None,
) -> dict[str, Any]:
    episodes = bundle["episodes"]
    y_chunks = [episodes[i]["y"] for i in split["train"] if i in episodes]
    y_train = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0, 4), np.float32)
    mean_action = y_train.mean(axis=0) if y_train.size else np.zeros(4, np.float32)
    ridge_x, ridge_y = stack_split(episodes, split["train"], max_rows=5_000)
    ridge_w = ridge_fit(ridge_x, ridge_y) if ridge_x.shape[0] > ridge_x.shape[1] else None

    def _eval_ids(ids: list[int], write_plots: bool) -> dict[str, Any]:
        gru_true: list[np.ndarray] = []
        gru_pred: list[np.ndarray] = []
        ridge_pred: list[np.ndarray] = []
        mean_pred: list[np.ndarray] = []
        per_episode: list[dict[str, Any]] = []
        for eid in ids:
            ep = episodes[eid]
            x, y = ep["x"], ep["y"]
            g = predict_gru_episode(model, x)
            m = np.broadcast_to(mean_action, y.shape).copy()
            r = ridge_predict(ridge_w, x) if ridge_w is not None else m
            gru_true.append(y)
            gru_pred.append(g)
            ridge_pred.append(r)
            mean_pred.append(m)
            per_episode.append(
                {
                    "episode_id": eid,
                    "n": int(y.shape[0]),
                    "gru": _metrics(y, g),
                }
            )
            if write_plots and plots_dir is not None and plt is not None:
                plots_dir.mkdir(parents=True, exist_ok=True)
                t = ep["sim_time_s"]
                fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                for i, name in enumerate(CONTROL_NAMES):
                    axes[i].plot(t, y[:, i], label="expert", color="#9fb0c3")
                    axes[i].plot(t, g[:, i], label="GRU", color="#7ad0a5")
                    axes[i].plot(t, r[:, i], label="Ridge", color="#e0b35a", alpha=0.8)
                    axes[i].set_ylabel(name)
                    axes[i].grid(True, alpha=0.2)
                axes[0].legend(loc="upper right")
                axes[0].set_title(f"held-out episode {eid} — decoder vs expert (not biological)")
                axes[-1].set_xlabel("sim time (s)")
                fig.tight_layout()
                fig.savefig(plots_dir / f"episode_{eid}_controls.png", dpi=120)
                plt.close(fig)
        yt = np.concatenate(gru_true, axis=0)
        return {
            "gru": _metrics(yt, np.concatenate(gru_pred, axis=0)),
            "ridge": _metrics(yt, np.concatenate(ridge_pred, axis=0)),
            "mean": _metrics(yt, np.concatenate(mean_pred, axis=0)),
            "episodes": per_episode,
        }

    test = _eval_ids(split["test"], write_plots=True)
    val = _eval_ids(split["val"], write_plots=False)
    gru_better = _gru_beats_ridge(test["gru"], test["ridge"])
    return {
        "val": val,
        "test": test,
        "gru_materially_improves_on_ridge": gru_better,
        "mean_action": mean_action.tolist(),
        "ridge_fit_rows": int(ridge_x.shape[0]),
        "plots_dir": str(plots_dir) if plots_dir else None,
    }


def _gru_beats_ridge(gru: dict[str, dict[str, float]], ridge: dict[str, dict[str, float]]) -> dict[str, Any]:
    better = {}
    for name in CONTROL_NAMES:
        g, r = gru[name], ridge[name]
        better[name] = {
            "mae_lower": g["mae"] < r["mae"],
            "pearson_higher": (g["pearson"] > r["pearson"])
            if np.isfinite(g["pearson"]) and np.isfinite(r["pearson"])
            else False,
        }
    n_mae = sum(1 for name in CONTROL_NAMES if better[name]["mae_lower"])
    return {"per_control": better, "mae_wins": n_mae, "material": n_mae >= 2}


def evaluate_checkpoint(
    checkpoint: Path,
    data: Path | Sequence[Path],
    *,
    plots_dir: Path | None = None,
    split: dict[str, list[int]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    artifact = DecoderArtifact.load(checkpoint)
    data_paths = [Path(data)] if isinstance(data, (str, Path)) else [Path(path) for path in data]
    bundle = load_compact_tables(data_paths)
    artifact.assert_dn_ordering(bundle["dn_body_ids"])
    if split is None:
        split_path = Path(checkpoint).with_name("split.json")
        if split_path.exists():
            split = {k: [int(x) for x in v] for k, v in json.loads(split_path.read_text()).items()}
        else:
            split = split_episode_ids(
                sorted(bundle["episodes"]),
                train=40,
                val=5,
                test=5,
                seed=seed,
            )
    required_ids = set(split["train"]) | set(split["val"]) | set(split["test"])
    missing_ids = sorted(required_ids - set(bundle["episodes"]))
    if missing_ids:
        raise ValueError(
            "the checkpoint split does not match the supplied dataset collection; "
            f"missing episode ids {missing_ids[:8]}. Pass every training dataset with repeated --data."
        )
    plots_dir = plots_dir or Path(checkpoint).with_name("plots")
    metrics = evaluate_splits(artifact.model, bundle, split, plots_dir=plots_dir)
    payload = {
        "checkpoint": str(checkpoint),
        "parameter_count": artifact.model.parameter_count(),
        "kind": artifact.kind,
        "biological_learning": False,
        "input_kind": artifact.model.config.input_kind if hasattr(artifact.model.config, "input_kind") else "dn_windowed_rates",
        "split": split,
        **metrics,
    }
    out = Path(checkpoint).with_name("offline_eval.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/decoder/best.pt"))
    parser.add_argument("--data", type=Path, action="append", dest="data_paths")
    parser.add_argument("--plots", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = evaluate_checkpoint(
        args.checkpoint,
        args.data_paths or [Path("data/decoder/expert_dn_controls.parquet")],
        plots_dir=args.plots,
        seed=args.seed,
    )
    print(json.dumps({"test": payload["test"]["gru"], "ridge": payload["test"]["ridge"], "mean": payload["test"]["mean"], "gru_vs_ridge": payload["gru_materially_improves_on_ridge"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
