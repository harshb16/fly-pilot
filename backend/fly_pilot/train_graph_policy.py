"""Imitation pretraining for the trainable MaleCNS population-graph policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.graph_policy import (
    OBSERVATION_FEATURES,
    ConnectomeGraphPolicy,
    GraphPolicyArtifact,
    GraphPolicyConfig,
    build_population_graph,
    default_graph_checkpoint_path,
    table_observation_matrix,
)

GUIDANCE_COLUMNS = (
    "roll_command_deg",
    "pitch_command_deg",
    "target_airspeed_kts",
    "throttle_trim",
)
REQUIRED_COLUMNS = {
    "episode_id",
    "phase",
    "sim_time_s",
    "along_m",
    "right_m",
    "alt_agl_m",
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
    *GUIDANCE_COLUMNS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        controls: np.ndarray,
        weights: np.ndarray,
        episodes: np.ndarray,
        selected: set[int],
        sequence_length: int,
        stride: int,
    ) -> None:
        self.features = features
        self.controls = controls
        self.weights = weights
        self.indices: list[tuple[int, int]] = []
        for episode_id in sorted(selected):
            rows = np.flatnonzero(episodes == episode_id)
            if not len(rows):
                continue
            start, stop = int(rows[0]), int(rows[-1]) + 1
            if stop - start <= sequence_length:
                self.indices.append((start, stop))
                continue
            for left in range(start, stop - sequence_length + 1, stride):
                self.indices.append((left, left + sequence_length))
            if self.indices[-1][1] != stop:
                self.indices.append((stop - sequence_length, stop))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = self.indices[index]
        return (
            torch.from_numpy(self.features[left:right]),
            torch.from_numpy(self.controls[left:right]),
            torch.from_numpy(self.weights[left:right]),
        )


def _normalize_guidance(columns: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.clip(np.asarray(columns["roll_command_deg"], dtype=np.float32) / 25.0, -1.0, 1.0),
            np.clip(np.asarray(columns["pitch_command_deg"], dtype=np.float32) / 8.0, -1.0, 1.0),
            np.clip((np.asarray(columns["target_airspeed_kts"], dtype=np.float32) - 65.0) / 15.0, -1.0, 1.0),
            np.clip(np.asarray(columns["throttle_trim"], dtype=np.float32) / 0.60, 0.0, 1.0),
        ]
    ).astype(np.float32)


def load_demonstrations(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    schema = pq.read_schema(path)
    missing = REQUIRED_COLUMNS - set(schema.names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table = pq.read_table(path, columns=sorted(REQUIRED_COLUMNS))
    columns = {name: table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in table.column_names if name != "phase"}
    columns["phase"] = np.asarray(table["phase"].combine_chunks().to_pylist(), dtype=object)
    features = table_observation_matrix(columns).astype(np.float32)
    controls = _normalize_guidance(columns)
    episodes = np.asarray(columns["episode_id"], dtype=np.int64)
    phase = columns["phase"]
    weights = np.ones(len(episodes), dtype=np.float32)
    weights[np.isin(phase, ["flare", "touchdown", "rollout"])] = 4.0
    return features, controls, weights, episodes


def split_episodes(episodes: np.ndarray, seed: int, val_fraction: float = 0.2) -> tuple[set[int], set[int]]:
    ids = np.unique(episodes).astype(int).tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return set(ids[n_val:]), set(ids[:n_val])


def _epoch(
    model: ConnectomeGraphPolicy,
    loader: DataLoader,
    target_scale: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for features, targets, sample_weights in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(features)
        per_step = (((prediction - targets) / target_scale) ** 2).mean(dim=-1)
        loss = (per_step * sample_weights).sum() / sample_weights.sum()
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        total += float(loss.detach()) * int(features.shape[0])
        count += int(features.shape[0])
    return total / max(count, 1)


def train_graph_policy(
    data: Path,
    output: Path,
    *,
    epochs: int = 40,
    patience: int = 8,
    seed: int = 17,
    batch_size: int = 32,
    sequence_length: int = 64,
    learning_rate: float = 8e-4,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features, controls, weights, episodes = load_demonstrations(data)
    train_ids, val_ids = split_episodes(episodes, seed)
    train_rows = np.isin(episodes, list(train_ids))
    mean = features[train_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features[train_rows].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    target_scale_np = np.maximum(controls[train_rows].std(axis=0), np.asarray([0.04, 0.04, 0.04, 0.08], dtype=np.float32))

    graph = build_population_graph(Connectome.load())
    config = GraphPolicyConfig()
    model = ConnectomeGraphPolicy(config, graph, mean=mean, std=std)
    train_set = SequenceDataset(features, controls, weights, episodes, train_ids, sequence_length, sequence_length // 2)
    val_set = SequenceDataset(features, controls, weights, episodes, val_ids, sequence_length, sequence_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    target_scale = torch.from_numpy(target_scale_np).view(1, 1, 4)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, train_loader, target_scale, optimizer)
        with torch.no_grad():
            val_loss = _epoch(model, val_loader, target_scale, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:02d} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    commit, dirty = _git_state()
    metadata = {
        "dataset": {"path": str(data), "sha256": _sha256(data), "rows": int(len(features))},
        "training": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "patience": patience,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "train_episode_ids": sorted(train_ids),
            "validation_episode_ids": sorted(val_ids),
            "target_scale": target_scale_np.astype(float).tolist(),
            "targets": {
                "roll": "roll_command_deg / 25",
                "pitch": "pitch_command_deg / 8",
                "airspeed": "(target_airspeed_kts - 65) / 15",
                "throttle_trim": "throttle_trim / 0.60",
            },
        },
        "source": {"git_commit": commit, "git_dirty": dirty},
        "metrics": {"best_validation_loss": best_loss, "history": history},
        "config": asdict(config),
    }
    artifact = GraphPolicyArtifact(config=config, graph=graph, model=model, metadata=metadata)
    artifact.save(output)
    report = {"checkpoint": str(output), "graph_sha256": graph.sha256, **metadata}
    output.with_name("train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=default_graph_checkpoint_path())
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args(argv)
    report = train_graph_policy(
        args.data,
        args.output,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
    )
    print(json.dumps({"checkpoint": report["checkpoint"], "best_validation_loss": report["metrics"]["best_validation_loss"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
