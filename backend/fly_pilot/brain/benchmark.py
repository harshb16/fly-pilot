"""CLI: python -m fly_pilot.brain.benchmark — measured full-network timings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fly_pilot.brain.config import LIFConfig, default_data_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.model import MaleCNSLIF


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    connectome = Connectome.load(data_dir)
    load_s = time.perf_counter() - t0
    rss_after_load = _rss_bytes()

    t1 = time.perf_counter()
    config = LIFConfig(seed=args.seed)
    net = MaleCNSLIF(connectome, config)
    init_s = time.perf_counter() - t1
    rss_after_init = _rss_bytes()

    for _ in range(args.warmup):
        net.step()

    t2 = time.perf_counter()
    n_spikes = 0
    for _ in range(args.steps):
        n_spikes += net.step().n_spikes
    step_s = time.perf_counter() - t2
    steps_per_s = args.steps / step_s if step_s else 0.0
    sim_s = args.steps * config.dt
    ratio = sim_s / step_s if step_s else 0.0

    report = {
        "host": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "nproc": os.cpu_count(),
        },
        "dataset": connectome.manifest.get("dataset"),
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "synapse_count_sum": connectome.synapse_count_sum,
        "counts_csr_bytes": connectome.counts_nbytes(),
        "runtime_csr_bytes": int(
            net.weights.data.nbytes + net.weights.indices.nbytes + net.weights.indptr.nbytes
        ),
        "runtime_csc_bytes": int(
            net.weights_csc.data.nbytes + net.weights_csc.indices.nbytes + net.weights_csc.indptr.nbytes
        ),
        "connectome_load_seconds": round(load_s, 4),
        "initialization_seconds": round(init_s, 4),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "timed_wall_seconds": round(step_s, 4),
        "timesteps_per_wall_second": round(steps_per_s, 2),
        "simulated_seconds": round(sim_s, 4),
        "simulated_per_wall": round(ratio, 4),
        "mean_spikes_per_step": round(n_spikes / args.steps, 2),
        "rss_bytes_before": rss_before,
        "rss_bytes_after_load": rss_after_load,
        "rss_bytes_after_init": rss_after_init,
        "dt": config.dt,
        "seed": config.seed,
        "note": (
            "Measured on this process, full un-pruned network, CSC event matvec. "
            "Not a theoretical peak. LIF dynamics are a modeling choice."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
