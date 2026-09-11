"""CLI: python -m fly_pilot.brain.benchmark — measured full-network timings.

Reports multiple firing regimes on the un-pruned MaleCNS graph. CSC event
propagation remains the default; a denser strategy is only noted if it would
win, and we do not drop edges to inflate steps/s.
"""

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
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.stimulation import PulseStimulation


def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _run_regime(
    net: MaleCNSLIF,
    *,
    name: str,
    steps: int,
    warmup: int,
    stimulation: PulseStimulation | None,
) -> dict:
    net.reset()
    for _ in range(warmup):
        net.step(stimulation)
    t0 = time.perf_counter()
    n_spikes = 0
    n_edges = 0
    max_spikes = 0
    for _ in range(steps):
        result = net.step(stimulation)
        n_spikes += result.n_spikes
        n_edges += result.n_outgoing_edges
        max_spikes = max(max_spikes, result.n_spikes)
    wall = time.perf_counter() - t0
    dt = net.config.dt
    return {
        "name": name,
        "timed_steps": steps,
        "warmup_steps": warmup,
        "timed_wall_seconds": round(wall, 4),
        "timesteps_per_wall_second": round(steps / wall, 2) if wall else 0.0,
        "simulated_seconds": round(steps * dt, 4),
        "simulated_per_wall": round((steps * dt) / wall, 4) if wall else 0.0,
        "mean_spikes_per_step": round(n_spikes / steps, 2),
        "max_spikes_per_step": max_spikes,
        "mean_outgoing_edges_per_step": round(n_edges / steps, 1),
        "rss_bytes_after": _rss_bytes(),
        "stimulation": None
        if stimulation is None
        else {
            "label": stimulation.label,
            "n_targets": int(stimulation.indices.size),
            "amplitude": stimulation.amplitude,
        },
    }


def run_suite(
    connectome: Connectome,
    *,
    steps: int,
    warmup: int,
    seed: int,
) -> dict:
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    config = LIFConfig(seed=seed)
    net = MaleCNSLIF(connectome, config)
    init_s = time.perf_counter() - t0
    rss_after_init = _rss_bytes()
    index = NeuronIndex(connectome)
    r1r6 = index.query(cell_type="R1-R6")
    rng = np.random.default_rng(seed)
    n_hot = max(int(0.12 * connectome.n_neurons), 1)
    hot = np.sort(rng.choice(connectome.n_neurons, size=n_hot, replace=False).astype(np.int32))

    regimes = [
        _run_regime(net, name="background_only", steps=steps, warmup=warmup, stimulation=None),
        _run_regime(
            net,
            name="r1r6_moderate",
            steps=steps,
            warmup=warmup,
            stimulation=PulseStimulation(r1r6, 0, 10**9, 0.40, "R1-R6 moderate"),
        ),
        _run_regime(
            net,
            name="r1r6_strong",
            steps=steps,
            warmup=warmup,
            stimulation=PulseStimulation(r1r6, 0, 10**9, 1.20, "R1-R6 strong"),
        ),
        _run_regime(
            net,
            name="synthetic_high_activity",
            steps=steps,
            warmup=warmup,
            stimulation=PulseStimulation(hot, 0, 10**9, 1.50, "12% random cells"),
        ),
    ]
    return {
        "host": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "nproc": os.cpu_count(),
        },
        "dataset": connectome.manifest.get("dataset"),
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "n_r1r6": int(r1r6.size),
        "initialization_seconds": round(init_s, 4),
        "rss_bytes_before": rss_before,
        "rss_bytes_after_init": rss_after_init,
        "dt": config.dt,
        "seed": seed,
        "regimes": regimes,
        "engine": "csc_event_propagation",
        "pruned": False,
        "note": (
            "Full un-pruned MaleCNS graph. CSC walks outgoing edges of neurons "
            "that spiked. High-activity regimes increase that walk; they do not "
            "remove synapses. LIF dynamics are a modeling choice."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="single background-only report (Milestone 3 shape)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    connectome = Connectome.load(data_dir)
    load_s = time.perf_counter() - t0
    rss_after_load = _rss_bytes()

    if args.legacy:
        config = LIFConfig(seed=args.seed)
        net = MaleCNSLIF(connectome, config)
        rss_after_init = _rss_bytes()
        for _ in range(args.warmup):
            net.step()
        t2 = time.perf_counter()
        n_spikes = 0
        for _ in range(args.steps):
            n_spikes += net.step().n_spikes
        step_s = time.perf_counter() - t2
        report = {
            "host": {"python": sys.version.split()[0], "numpy": np.__version__, "nproc": os.cpu_count()},
            "n_neurons": connectome.n_neurons,
            "n_edges": connectome.n_edges,
            "connectome_load_seconds": round(load_s, 4),
            "initialization_seconds": 0.0,
            "timed_steps": args.steps,
            "timed_wall_seconds": round(step_s, 4),
            "timesteps_per_wall_second": round(args.steps / step_s, 2) if step_s else 0.0,
            "simulated_per_wall": round((args.steps * config.dt) / step_s, 4) if step_s else 0.0,
            "mean_spikes_per_step": round(n_spikes / args.steps, 2),
            "rss_bytes_before": rss_before,
            "rss_bytes_after_load": rss_after_load,
            "rss_bytes_after_init": rss_after_init,
            "dt": config.dt,
            "seed": config.seed,
        }
    else:
        report = run_suite(connectome, steps=args.steps, warmup=args.warmup, seed=args.seed)
        report["connectome_load_seconds"] = round(load_s, 4)
        report["rss_bytes_after_load"] = rss_after_load

    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
