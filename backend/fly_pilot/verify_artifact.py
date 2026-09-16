"""Verify the committed decoder checkpoint without training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fly_pilot.brain.config import default_data_dir, prepared_dir
from fly_pilot.brain.decoder import DecoderArtifact, default_checkpoint_path, sha256_path


def verify_decoder_artifact(
    path: Path,
    *,
    require_prepared: bool = False,
    check_prepared: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_suffix(".meta.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"decoder metadata sidecar not found: {sidecar}")
    meta = json.loads(sidecar.read_text())
    digest = sha256_path(path)
    expected_digest = str(meta.get("checkpoint_sha256") or "")
    if not expected_digest or digest != expected_digest:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: got {digest}, expected {expected_digest or 'missing'}"
        )
    required = (
        "git_commit",
        "git_dirty",
        "training_command",
        "dependencies",
        "seed_ranges",
        "dataset",
        "metrics",
    )
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"artifact metadata is missing {missing}")
    artifact = DecoderArtifact.load(path)
    body_ids = np.asarray(artifact.dn_body_ids, dtype=np.int64)
    if body_ids.size != artifact.config.n_descending:
        raise ValueError("checkpoint DN body-id count does not match decoder config")
    if np.unique(body_ids).size != body_ids.size:
        raise ValueError("checkpoint DN body-id ordering contains duplicates")

    prepared = prepared_dir(default_data_dir())
    has_prepared = (prepared / "manifest.json").exists() and (prepared / "weights.npz").exists()
    order_check = "embedded-structural"
    if has_prepared and check_prepared:
        from fly_pilot.brain.observing import ObservingMaleCNS

        observer = ObservingMaleCNS.load()
        artifact.assert_dn_ordering(observer.dn.spec.body_ids)
        order_check = "prepared-malecns-exact"
    elif require_prepared:
        raise FileNotFoundError("prepared MaleCNS data is required for an exact DN ordering check")

    features = np.zeros(artifact.config.input_dim, dtype=np.float32)
    artifact.model.reset_state()
    first = artifact.model.step_numpy(features)
    artifact.model.reset_state()
    repeated = artifact.model.step_numpy(features)
    if not np.allclose(first, repeated, atol=1e-7):
        raise ValueError("decoder reset is not deterministic")
    if not (
        np.all(np.isfinite(first))
        and np.all(first[:3] >= -1.0)
        and np.all(first[:3] <= 1.0)
        and 0.0 <= float(first[3]) <= 1.0
    ):
        raise ValueError("decoder smoke output is non-finite or outside legal control bounds")
    return {
        "ok": True,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "parameter_count": artifact.model.parameter_count(),
        "input_dim": artifact.config.input_dim,
        "dn_n": artifact.config.n_descending,
        "dn_order_check": order_check,
        "training_data_required": False,
        "smoke_controls": [float(x) for x in first.tolist()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = verify_decoder_artifact(
        args.checkpoint or default_checkpoint_path(),
        require_prepared=args.require_prepared,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
