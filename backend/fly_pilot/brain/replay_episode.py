"""Replay a recorded observing-expert episode without Three.js or JSBSim.

    python -m fly_pilot.brain.replay_episode data/observing/expert_observing.parquet

Loads retinal currents from the dataset, resets MaleCNS with the recorded
seed/config, and checks that the spike-checksum sequence matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fly_pilot.brain.config import LIFConfig, default_data_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.observing import ObservingMaleCNS
from fly_pilot.brain.stimulation import CurrentStimulation
from fly_pilot.record_observing import synthetic_observer


def _lif_from_meta(meta: dict) -> LIFConfig:
    lif = meta.get("observer", {}).get("lif", {})
    return LIFConfig(
        dt=float(lif.get("dt", 0.02)),
        tau_m=float(lif.get("tau_m", 0.1)),
        tonic_current=float(lif.get("tonic_current", 0.18)),
        synaptic_gain=float(lif.get("synaptic_gain", 1.5)),
        background_rate_hz=float(lif.get("background_rate_hz", 1.2)),
        background_amplitude=float(lif.get("background_amplitude", 0.22)),
        normalize_incoming=bool(lif.get("normalize_incoming", True)),
        seed=int(lif.get("seed", 64)),
    )


def replay_table(
    parquet_path: Path,
    *,
    observer: ObservingMaleCNS | None = None,
    meta: dict | None = None,
    episode_id: int | None = None,
) -> dict:
    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    sidecar = parquet_path.with_suffix(".meta.json")
    if meta is None and sidecar.exists():
        meta = json.loads(sidecar.read_text())
    meta = meta or {}
    synthetic = bool(meta.get("synthetic"))
    graph_seed = meta.get("synthetic_graph_seed")
    if observer is None:
        if synthetic:
            observer = synthetic_observer(
                seed=int(graph_seed if graph_seed is not None else 64)
            )
        else:
            observer = ObservingMaleCNS(
                Connectome.load(default_data_dir()),
                config=_lif_from_meta(meta),
            )

    n_rows = len(data["neural_step"])
    matches = 0
    compared = 0
    mismatches: list[dict] = []
    current_episode = None
    for i in range(n_rows):
        ep = int(data["episode_id"][i])
        if episode_id is not None and ep != episode_id:
            continue
        neural_step = int(data["neural_step"][i])
        if neural_step < 0:
            continue
        if current_episode != ep:
            current_episode = ep
            observer.reset(seed=int(data["seed"][i]))
            seen_steps: set[int] = set()
        if neural_step in seen_steps:
            continue
        seen_steps.add(neural_step)
        blob = data["retinal_currents_f16"][i]
        currents = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        if currents.size != observer.mapping.n_receptors:
            raise ValueError(
                f"row {i}: retinal length {currents.size} != mapped receptors "
                f"{observer.mapping.n_receptors}"
            )
        stim = CurrentStimulation(observer.mapping.indices, currents, label="replay")
        result = observer.net.step(stim)
        recorded = str(data["spike_checksum"][i])
        compared += 1
        if result.checksum == recorded:
            matches += 1
        else:
            mismatches.append(
                {
                    "row": i,
                    "episode_id": ep,
                    "neural_step": neural_step,
                    "recorded": recorded,
                    "replayed": result.checksum,
                    "n_spikes_recorded": int(data["n_spikes"][i]),
                    "n_spikes_replayed": result.n_spikes,
                }
            )
            if len(mismatches) >= 8:
                break
    ok = compared > 0 and matches == compared
    return {
        "path": str(parquet_path),
        "compared": compared,
        "matches": matches,
        "ok": ok,
        "mismatches": mismatches,
        "synthetic": synthetic,
        "n_receptors": observer.mapping.n_receptors,
        "n_neurons": observer.connectome.n_neurons,
        "note": (
            "Replay applies recorded retinal currents to MaleCNS with the recorded "
            "seed. Identical checksums are required on the same platform / NumPy."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = replay_table(args.dataset, episode_id=args.episode)
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
