"""Train the causal GRU decoder on compact DN-rate demonstrations.

Episode-level splits only. The learned weights are an external decoder;
MaleCNS synapses are not updated.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from fly_pilot.brain.decoder import (
    CONTROL_NAMES,
    CausalTemporalDecoder,
    DecoderArtifact,
    DecoderConfig,
    sha256_path,
)

SCHEMA_VERSION = "decoder-training-v1"
PORTFOLIO_SEED_RANGES = {
    "training": {"start": 1000},
    "validation": {"start": 3000, "episodes": 20},
    "test": {"start": 4000, "episodes": 100},
}


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
    except Exception:
        return True


def dependency_versions() -> dict[str, str]:
    packages = ("jsbsim", "numpy", "scipy", "websockets", "pyarrow", "torch", "matplotlib")
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def decode_rates(blob: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    if arr.size != n:
        raise ValueError(f"rate blob length {arr.size} != n_descending {n}")
    return arr


def load_compact_table(path: Path) -> dict[str, Any]:
    path = Path(path)
    table = pq.read_table(path)
    cols = set(table.column_names)
    required = {
        "episode_id",
        "timestep",
        "sim_time_s",
        "dn_rates_100_f16",
        "dn_rates_260_f16",
        "expert_aileron",
        "expert_elevator",
        "expert_rudder",
        "expert_throttle",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    forbidden = {
        "along_m",
        "right_m",
        "alt_agl_m",
        "airspeed_kts",
        "pitch_deg",
        "roll_deg",
        "heading_deg",
        "retinal_currents_f16",
    }
    present_forbidden = sorted(forbidden & cols)
    if present_forbidden:
        raise ValueError(f"{path} includes non-decoder fields {present_forbidden}")
    sidecar = path.with_suffix(".meta.json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    dn_n = int(meta.get("dn_n") or 0)
    if dn_n <= 0:
        sample = table["dn_rates_100_f16"][0].as_py()
        dn_n = int(np.frombuffer(sample, dtype=np.float16).size)
    episodes: dict[int, dict[str, list]] = {}
    for i in range(table.num_rows):
        eid = int(table["episode_id"][i].as_py())
        bucket = episodes.setdefault(
            eid,
            {"t": [], "x": [], "y": [], "sim": []},
        )
        short = decode_rates(table["dn_rates_100_f16"][i].as_py(), dn_n)
        long = decode_rates(table["dn_rates_260_f16"][i].as_py(), dn_n)
        x = np.concatenate([short, long], axis=0)
        y = np.array(
            [
                float(table["expert_aileron"][i].as_py()),
                float(table["expert_elevator"][i].as_py()),
                float(table["expert_rudder"][i].as_py()),
                float(table["expert_throttle"][i].as_py()),
            ],
            dtype=np.float32,
        )
        bucket["t"].append(int(table["timestep"][i].as_py()))
        bucket["sim"].append(float(table["sim_time_s"][i].as_py()))
        bucket["x"].append(x)
        bucket["y"].append(y)
    packed = {}
    for eid, bucket in episodes.items():
        order = np.argsort(np.asarray(bucket["t"]))
        packed[eid] = {
            "timestep": np.asarray(bucket["t"], dtype=np.int32)[order],
            "sim_time_s": np.asarray(bucket["sim"], dtype=np.float64)[order],
            "x": np.stack([bucket["x"][j] for j in order], axis=0),
            "y": np.stack([bucket["y"][j] for j in order], axis=0),
        }
    return {
        "episodes": packed,
        "dn_n": dn_n,
        "dn_body_ids": np.asarray(meta.get("dn_body_ids") or [], dtype=np.int64),
        "meta": meta,
        "path": str(path),
        "rows": table.num_rows,
        "n_episodes": len(packed),
        "sha256": sha256_path(path),
        "episode_sources": {
            int(eid): {"path": str(path), "source_episode_id": int(eid)}
            for eid in packed
        },
    }


def load_compact_tables(paths: Sequence[Path]) -> dict[str, Any]:
    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("at least one decoder dataset is required")
    bundles = [load_compact_table(path) for path in normalized]
    if len(bundles) == 1:
        one = dict(bundles[0])
        one["datasets"] = [
            {"path": one["path"], "sha256": one["sha256"], "rows": one["rows"]}
        ]
        return one
    first = bundles[0]
    expected_n = int(first["dn_n"])
    expected_ids = np.asarray(first["dn_body_ids"], dtype=np.int64)
    episodes: dict[int, dict[str, np.ndarray]] = {}
    sources: dict[int, dict[str, Any]] = {}
    next_id = 0
    for bundle in bundles:
        if int(bundle["dn_n"]) != expected_n:
            raise ValueError("decoder datasets use different descending-neuron counts")
        ids = np.asarray(bundle["dn_body_ids"], dtype=np.int64)
        if expected_ids.size and ids.size and not np.array_equal(ids, expected_ids):
            raise ValueError("decoder datasets use different DN body-id ordering")
        for local_id in sorted(bundle["episodes"]):
            episodes[next_id] = bundle["episodes"][local_id]
            sources[next_id] = {
                "path": bundle["path"],
                "source_episode_id": int(local_id),
            }
            next_id += 1
    return {
        "episodes": episodes,
        "dn_n": expected_n,
        "dn_body_ids": expected_ids,
        "meta": {"combined": True},
        "path": [bundle["path"] for bundle in bundles],
        "rows": int(sum(bundle["rows"] for bundle in bundles)),
        "n_episodes": len(episodes),
        "sha256": None,
        "episode_sources": sources,
        "datasets": [
            {"path": bundle["path"], "sha256": bundle["sha256"], "rows": bundle["rows"]}
            for bundle in bundles
        ],
    }


def split_episode_ids(episode_ids: list[int], *, train: int, val: int, test: int, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(episode_ids), dtype=np.int64)
    rng.shuffle(ids)
    ids = [int(x) for x in ids.tolist()]
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": ids, "val": [], "test": []}
    if n == 2:
        return {"train": ids[:1], "val": [], "test": ids[1:]}
    n_test = min(test, max(1, n // 10)) if n >= 10 else min(test, 1)
    n_val = min(val, max(1, n // 10)) if n >= 10 else min(val, 1)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = 1 if n > 2 else 0
        n_test = n - n_train - n_val
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


class SequenceDataset(Dataset):
    def __init__(
        self,
        episodes: dict[int, dict[str, np.ndarray]],
        ids: list[int],
        seq_len: int,
        stride: int,
        *,
        balance_transients: bool = True,
    ) -> None:
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        opening = int(15.0 / 0.02)  # first 15 s of neural time
        flare = int(20.0 / 0.02)
        for eid in ids:
            x = episodes[eid]["x"]
            y = episodes[eid]["y"]
            t = x.shape[0]
            if t < seq_len:
                continue
            last = t - seq_len
            for start in range(0, last + 1, max(1, stride)):
                if balance_transients:
                    end = start + seq_len
                    opening_window = start < opening
                    flare_window = end > t - flare
                    cruise = not opening_window and not flare_window
                    if cruise and (start // max(1, stride)) % 5 != 0:
                        continue
                self.xs.append(x[start : start + seq_len])
                self.ys.append(y[start : start + seq_len])
            if last > 0 and last % max(1, stride) != 0:
                self.xs.append(x[-seq_len:])
                self.ys.append(y[-seq_len:])

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.xs[idx]), torch.from_numpy(self.ys[idx])


def fit_scaler(episodes: dict[int, dict[str, np.ndarray]], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    chunks = [episodes[i]["x"] for i in ids if i in episodes]
    stacked = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 1), np.float32)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def huber_per_control(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.08) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return loss.mean(dim=(0, 1))


CONTROL_LOSS_WEIGHTS = torch.tensor([2.4, 1.8, 1.4, 1.0])  # aileron, elevator, rudder, throttle


def init_control_heads(model: CausalTemporalDecoder, mean: np.ndarray) -> None:
    """Start near the mean-action baseline so the GRU learns residuals."""
    mean = np.asarray(mean, dtype=np.float64).reshape(4)
    with torch.no_grad():
        for i, head in enumerate((model.head_aileron, model.head_elevator, model.head_rudder)):
            m = float(np.clip(mean[i], -0.95, 0.95))
            bias = 0.5 * np.log((1.0 + m) / max(1.0 - m, 1e-6))
            head.bias.fill_(bias)
            head.weight.mul_(0.05)
        m = float(np.clip(mean[3], 0.02, 0.98))
        model.head_throttle.bias.fill_(float(np.log(m / (1.0 - m))))
        model.head_throttle.weight.mul_(0.05)


def timestep_weights(n: int, dt: float = 0.02) -> torch.Tensor:
    """Upweight opening corrections and flare; cruise still contributes."""
    t = np.arange(n, dtype=np.float32) * float(dt)
    w = np.ones(n, dtype=np.float32)
    duration = float(n * dt)
    w[t < 20.0] *= 3.0
    w[t > max(duration - 25.0, 0.0)] *= 2.5
    return torch.from_numpy(w)


def aligned_xy(ep: dict[str, np.ndarray], target_shift: int) -> tuple[np.ndarray, np.ndarray]:
    x = ep["x"]
    y = ep["y"]
    if target_shift > 0 and x.shape[0] > target_shift + 2:
        return x[:-target_shift], y[target_shift:]
    return x, y


def batch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth_weight: float,
    target_mean: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_elem = nn.functional.smooth_l1_loss(pred, target, beta=0.08, reduction="none")
    if target_mean is not None:
        emphasis = 1.0 + 6.0 * (target - target_mean).abs()
        per_elem = per_elem * emphasis
    if time_weights is not None:
        tw = time_weights.to(device=pred.device, dtype=pred.dtype).reshape(1, -1, 1)
        per_elem = per_elem * tw
    weights = CONTROL_LOSS_WEIGHTS.to(device=pred.device, dtype=pred.dtype)
    per = per_elem.mean(dim=(0, 1)) * weights
    huber = per.mean()
    delta = pred[:, 1:] - pred[:, :-1]
    smooth = (delta.square().mean()) if pred.shape[1] > 1 else pred.new_zeros(())
    total = huber + smooth_weight * smooth
    stats = {f"huber_{name}": float(per[i].item()) for i, name in enumerate(CONTROL_NAMES)}
    stats["huber"] = float(huber.item())
    stats["smooth"] = float(smooth.item()) if torch.is_tensor(smooth) else 0.0
    stats["loss"] = float(total.item())
    return total, stats


def _accumulate(totals: dict[str, float], stats: dict[str, float], scale: float = 1.0) -> None:
    for k, v in stats.items():
        totals[k] = totals.get(k, 0.0) + v * scale


def run_tbptt_epoch(
    model: CausalTemporalDecoder,
    bundle: dict[str, Any],
    ids: list[int],
    optimizer: torch.optim.Optimizer | None,
    *,
    chunk_len: int,
    smooth_weight: float,
    target_mean: torch.Tensor,
    target_shift: int,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Train/eval with persistent GRU state across each episode (matches flight)."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_chunks = 0
    order = list(ids)
    if rng is not None and training:
        rng.shuffle(order)
    for eid in order:
        x_np, y_np = aligned_xy(bundle["episodes"][eid], target_shift)
        if x_np.shape[0] < 4:
            continue
        x = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(y_np)).unsqueeze(0)
        tw = timestep_weights(int(x.shape[1]))
        hidden: torch.Tensor | None = None
        t = 0
        T = int(x.shape[1])
        while t < T:
            end = min(t + max(1, chunk_len), T)
            xb = x[:, t:end]
            yb = y[:, t:end]
            wb = tw[t:end]
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred, hidden = model(xb, hidden)
            loss, stats = batch_loss(
                pred, yb, smooth_weight, target_mean=target_mean, time_weights=wb
            )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                hidden = hidden.detach()
            else:
                hidden = hidden.detach()
            _accumulate(totals, stats)
            n_chunks += 1
            t = end
    if n_chunks == 0:
        return {"loss": float("inf")}
    return {k: v / n_chunks for k, v in totals.items()}


def count_chunks(bundle: dict[str, Any], ids: list[int], chunk_len: int, target_shift: int) -> int:
    n = 0
    for eid in ids:
        x, _ = aligned_xy(bundle["episodes"][eid], target_shift)
        t = int(x.shape[0])
        if t < 4:
            continue
        n += int(np.ceil(t / max(1, chunk_len)))
    return n


def train_decoder(
    data_path: Path | Sequence[Path],
    output: Path,
    *,
    seed: int = 0,
    epochs: int = 40,
    patience: int = 8,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    seq_len: int = 50,
    stride: int = 10,
    smooth_weight: float = 0.01,
    train_episodes: int = 40,
    val_episodes: int = 5,
    test_episodes: int = 5,
    gru_hidden: int = 128,
    hidden_linear: int = 256,
    dropout: float = 0.1,
    target_shift: int = 2,
    training_command: Sequence[str] | None = None,
    seed_ranges: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del batch_size, stride  # windowed loader is no longer the training path
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    data_paths = [Path(data_path)] if isinstance(data_path, (str, Path)) else [Path(p) for p in data_path]
    bundle = load_compact_tables(data_paths)
    ids = sorted(bundle["episodes"])
    split = split_episode_ids(ids, train=train_episodes, val=val_episodes, test=test_episodes, seed=seed)
    mean, std = fit_scaler(bundle["episodes"], split["train"])
    config = DecoderConfig(
        n_descending=int(bundle["dn_n"]),
        gru_hidden=gru_hidden,
        hidden_linear=hidden_linear,
        dropout=dropout,
        sequence_length=seq_len,
    )
    model = CausalTemporalDecoder(config, mean=mean, std=std)
    y_train = np.concatenate([bundle["episodes"][i]["y"] for i in split["train"]], axis=0)
    target_mean_np = y_train.mean(axis=0).astype(np.float32)
    init_control_heads(model, target_mean_np)
    target_mean = torch.from_numpy(target_mean_np)
    n_train_sequences = count_chunks(bundle, split["train"], seq_len, target_shift)
    n_val_sequences = count_chunks(bundle, split["val"], seq_len, target_shift)
    if n_train_sequences == 0:
        raise RuntimeError("no training sequences — collect more episodes")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_stats = run_tbptt_epoch(
            model,
            bundle,
            split["train"],
            optimizer,
            chunk_len=seq_len,
            smooth_weight=smooth_weight,
            target_mean=target_mean,
            target_shift=target_shift,
            rng=rng,
        )
        if split["val"]:
            val_stats = run_tbptt_epoch(
                model,
                bundle,
                split["val"],
                None,
                chunk_len=seq_len,
                smooth_weight=smooth_weight,
                target_mean=target_mean,
                target_shift=target_shift,
            )
        else:
            val_stats = dict(train_stats)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        val_loss = float(val_stats["loss"])
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:02d}  train {train_stats['loss']:.4f}  "
            f"val {val_loss:.4f}  best {best_val:.4f}",
            flush=True,
        )
        if stale >= patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    from fly_pilot.evaluate_decoder import evaluate_splits

    plots_dir = output.parent / "plots"
    metrics = evaluate_splits(model, bundle, split, plots_dir=plots_dir)
    n_train_rows = int(sum(bundle["episodes"][i]["x"].shape[0] for i in split["train"]))
    artifact = DecoderArtifact(
        config=config,
        model=model,
        dn_body_ids=bundle["dn_body_ids"],
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        training_seed=seed,
        training_command=list(training_command or sys.argv),
        dependencies=dependency_versions(),
        seed_ranges=dict(seed_ranges or PORTFOLIO_SEED_RANGES),
        dataset={
            "paths": [str(path) for path in data_paths],
            "files": bundle["datasets"],
            "schema": SCHEMA_VERSION,
            "rows": bundle["rows"],
            "n_episodes": bundle["n_episodes"],
            "split": split,
            "episode_sources": {str(k): v for k, v in bundle["episode_sources"].items()},
            "sequence_length": seq_len,
            "training": "episode_tbptt",
            "target_shift_steps": target_shift,
            "n_train_rows": n_train_rows,
            "n_train_sequences": n_train_sequences,
            "n_val_sequences": n_val_sequences,
        },
        metrics={
            "best_val_loss": best_val,
            "history": history,
            "offline": metrics,
        },
    )
    output = Path(output)
    artifact.save(output)
    split_path = output.with_name("split.json")
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    report = {
        "checkpoint": str(output),
        "parameter_count": model.parameter_count(),
        "split": split,
        "n_train_sequences": n_train_sequences,
        "n_val_sequences": n_val_sequences,
        "n_train_rows": n_train_rows,
        "best_val_loss": best_val,
        "offline": metrics,
        "git_commit": artifact.git_commit,
        "git_dirty": artifact.git_dirty,
        "training_command": artifact.training_command,
        "dependencies": artifact.dependencies,
        "seed_ranges": artifact.seed_ranges,
        "datasets": bundle["datasets"],
        "seed": seed,
        "training": "episode_tbptt",
        "target_shift_steps": target_shift,
    }
    output.with_name("train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        dest="data_paths",
        type=Path,
        action="append",
        help="Compact decoder dataset; repeat for expert and DAgger files.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/decoder/best.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--hidden-linear", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-shift", type=int, default=2)
    args = parser.parse_args(argv)
    report = train_decoder(
        args.data_paths or [Path("data/decoder/expert_dn_controls.parquet")],
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seq_len=args.seq_len,
        stride=args.stride,
        smooth_weight=args.smooth_weight,
        gru_hidden=args.gru_hidden,
        hidden_linear=args.hidden_linear,
        dropout=args.dropout,
        target_shift=args.target_shift,
        training_command=[sys.executable, "-m", "fly_pilot.train_decoder", *sys.argv[1:]],
        seed_ranges=PORTFOLIO_SEED_RANGES,
    )
    print(json.dumps({k: report[k] for k in report if k != "offline"}, indent=2))
    print(json.dumps(report["offline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
