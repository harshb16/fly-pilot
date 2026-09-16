"""Verify the portable hybrid-guidance checkpoint without training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from fly_pilot.brain.graph_policy import (
    OBSERVATION_FEATURES,
    GraphPolicyArtifact,
    build_population_graph,
    default_graph_checkpoint_path,
)


def verify_graph_artifact(path: Path, *, compare_prepared: bool = False) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"graph checkpoint and metadata are required: {path}, {sidecar}")
    meta = json.loads(sidecar.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta.get("checkpoint_sha256"):
        raise ValueError("graph checkpoint SHA-256 mismatch")
    required = (
        "population_graph_sha256",
        "observation_features",
        "parameter_count",
        "uses_aircraft_telemetry",
        "biological_learning",
        "dataset",
        "training",
        "source",
        "metrics",
    )
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"graph artifact metadata is missing {missing}")
    artifact = GraphPolicyArtifact.load(path)
    if artifact.graph.sha256 != meta["population_graph_sha256"]:
        raise ValueError("embedded population graph hash differs from metadata")
    if tuple(meta["observation_features"]) != OBSERVATION_FEATURES:
        raise ValueError("observation feature ordering mismatch")
    graph_check = "embedded-structural"
    if compare_prepared:
        from fly_pilot.brain.connectome import Connectome

        prepared = build_population_graph(Connectome.load())
        if prepared.sha256 != artifact.graph.sha256:
            raise ValueError("prepared MaleCNS population graph differs from checkpoint")
        graph_check = "prepared-malecns-exact"
    zeros = np.zeros(artifact.config.observation_dim, dtype=np.float32)
    artifact.model.reset_state()
    first = artifact.model.step_numpy(zeros)
    artifact.model.reset_state()
    second = artifact.model.step_numpy(zeros)
    if not np.allclose(first, second, atol=1e-7):
        raise ValueError("graph policy reset is not deterministic")
    if not np.all(np.isfinite(first)):
        raise ValueError("graph policy smoke output is non-finite")
    return {
        "ok": True,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "population_graph_sha256": artifact.graph.sha256,
        "population_count": len(artifact.graph.names),
        "parameter_count": artifact.model.parameter_count(),
        "graph_check": graph_check,
        "training_data_required": False,
        "smoke_guidance_normalized": [float(x) for x in first.tolist()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_graph_checkpoint_path())
    parser.add_argument("--compare-prepared", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = verify_graph_artifact(args.checkpoint, compare_prepared=args.compare_prepared)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
