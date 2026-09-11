"""CLI: python -m fly_pilot.brain.demo — standalone full-network stimulation.

Does not start JSBSim or the browser. Requires prepared MaleCNS data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fly_pilot.brain.config import LIFConfig, default_data_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.model import MaleCNSLIF
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.replay import SpikeTrace
from fly_pilot.brain.stimulation import PulseStimulation
from fly_pilot.brain.telemetry import rate_hz


def _mean_rates(rows: list[dict[str, float]], name: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row.get(name, 0.0) for row in rows]))


def run_experiment(
    connectome: Connectome,
    *,
    config: LIFConfig,
    baseline_steps: int,
    stim_steps: int,
    recovery_steps: int,
    stim_amplitude: float,
    recurrent_enabled: bool,
    trace_path: Path | None,
) -> dict:
    index = NeuronIndex(connectome)
    photoreceptors = index.query(cell_type="R1-R6")
    lamina_l1 = index.query(type="L1")
    descending = index.query(superclass="descending_neuron")
    giant_fiber = index.query(cell_type="DNp01")
    if photoreceptors.size == 0:
        raise RuntimeError("R1-R6 photoreceptors were not found in the prepared annotations")

    populations = {
        "R1-R6": photoreceptors,
        "L1": lamina_l1,
        "descending_neuron": descending,
        "DNp01": giant_fiber,
    }
    stim = PulseStimulation(
        indices=photoreceptors,
        start_step=baseline_steps,
        end_step=baseline_steps + stim_steps,
        amplitude=stim_amplitude,
        label="R1-R6 synthetic current",
    )
    net = MaleCNSLIF(connectome, config, recurrent_enabled=recurrent_enabled)
    trace = SpikeTrace(
        seed=config.seed,
        n_neurons=connectome.n_neurons,
        n_steps=baseline_steps + stim_steps + recovery_steps,
        dt=config.dt,
        stimulation={
            "population": "R1-R6 photoreceptors (type/flywireType)",
            "n_cells": int(photoreceptors.size),
            "start_step": stim.start_step,
            "end_step": stim.end_step,
            "amplitude": stim_amplitude,
            "recurrent_enabled": recurrent_enabled,
        },
    )

    phase_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    phase_spikes: dict[str, int] = defaultdict(int)

    def phase_of(step: int) -> str:
        if step < baseline_steps:
            return "baseline"
        if step < baseline_steps + stim_steps:
            return "stimulation"
        return "recovery"

    t0 = time.perf_counter()
    for _ in range(trace.n_steps):
        result = net.step(stim)
        tel = net.telemetry(result, populations=populations, stimulated=photoreceptors)
        trace.record(result.fired)
        phase = phase_of(tel.step)
        phase_spikes[phase] += tel.n_spikes
        phase_rows[phase].append(tel.population_rates_hz)
    elapsed = time.perf_counter() - t0

    def summary(phase: str) -> dict:
        rows = phase_rows[phase]
        steps = max(len(rows), 1)
        out = {
            "steps": len(rows),
            "total_spikes": int(phase_spikes[phase]),
            "mean_global_rate_hz": rate_hz(phase_spikes[phase], connectome.n_neurons, config.dt * steps),
        }
        for name in populations:
            out[f"mean_{name}_hz"] = _mean_rates(rows, name)
        return out

    report = {
        "dataset": connectome.manifest.get("dataset"),
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "synapse_count_sum": connectome.synapse_count_sum,
        "recurrent_enabled": recurrent_enabled,
        "config": {
            "dt": config.dt,
            "tau_m": config.tau_m,
            "tonic_current": config.tonic_current,
            "synaptic_gain": config.synaptic_gain,
            "background_rate_hz": config.background_rate_hz,
            "background_amplitude": config.background_amplitude,
            "seed": config.seed,
            "normalize_incoming": config.normalize_incoming,
        },
        "populations": {name: int(idx.size) for name, idx in populations.items()},
        "stimulation": trace.stimulation,
        "baseline": summary("baseline"),
        "stimulation_phase": summary("stimulation"),
        "recovery": summary("recovery"),
        "wall_seconds": round(elapsed, 4),
        "timesteps_per_wall_second": round(trace.n_steps / elapsed, 2) if elapsed else 0.0,
        "final_spike_checksum": trace.final_checksum,
        "integrity": (
            "MaleCNS-derived spiking-network simulation using the measured connectome. "
            "LIF dynamics are a modeling choice, not validated fly physiology. "
            "Activity propagation is not biological validation."
        ),
    }
    r1_extra = (
        report["stimulation_phase"]["mean_R1-R6_hz"] - report["baseline"]["mean_R1-R6_hz"]
    ) * photoreceptors.size * config.dt * stim_steps
    global_extra = report["stimulation_phase"]["total_spikes"] - report["baseline"]["total_spikes"]
    report["evidence"] = {
        "stim_entered_R1R6": report["stimulation_phase"]["mean_R1-R6_hz"]
        > report["baseline"]["mean_R1-R6_hz"] + 5.0,
        "l1_rate_delta_hz": report["stimulation_phase"]["mean_L1_hz"] - report["baseline"]["mean_L1_hz"],
        "global_rate_rose": report["stimulation_phase"]["mean_global_rate_hz"]
        > report["baseline"]["mean_global_rate_hz"],
        "extra_spikes_beyond_targets": float(global_extra - r1_extra),
        "note": (
            "R1-R6 are modeled as histaminergic / inhibitory onto lamina in this "
            "approximation, so L1 rate may fall during photoreceptor stimulation. "
            "That is a signed-weight modeling effect, not a biological validation."
        ),
    }
    if trace_path is not None:
        trace.write(trace_path)
        report["trace_path"] = str(trace_path)
    return report


def _print_report(report: dict) -> None:
    print("MaleCNS-derived spiking-network simulation using the measured connectome.")
    print("LIF dynamics are a modeling choice — not a living fly and not validated physiology.")
    print()
    print(f"dataset:     {report['dataset']}")
    print(f"neurons:     {report['n_neurons']:,}")
    print(f"edges:       {report['n_edges']:,}  (aggregated synapse-count weights)")
    print(f"synapse Σ:   {report['synapse_count_sum']:,}")
    print(f"recurrent:   {report['recurrent_enabled']}")
    print(f"stim:        {report['stimulation']}")
    print(f"populations: {report['populations']}")
    for phase in ("baseline", "stimulation_phase", "recovery"):
        block = report[phase]
        print(f"\n{phase}:")
        for key, value in block.items():
            if isinstance(value, float):
                print(f"  {key:24s} {value:10.4f}")
            else:
                print(f"  {key:24s} {value}")
    print(f"\nwall seconds:          {report['wall_seconds']}")
    print(f"timesteps / wall s:    {report['timesteps_per_wall_second']}")
    print(f"replay checksum:       {report['final_spike_checksum']}")
    print(f"evidence:              {report['evidence']}")
    print(f"\n{report['integrity']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--baseline-steps", type=int, default=50)
    parser.add_argument("--stim-steps", type=int, default=50)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--stim-amplitude", type=float, default=1.2)
    parser.add_argument("--no-recurrent", action="store_true")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    t_load = time.perf_counter()
    connectome = Connectome.load(data_dir)
    load_s = time.perf_counter() - t_load
    print(
        f"Loaded {connectome.n_neurons:,} neurons / {connectome.n_edges:,} edges "
        f"in {load_s:.2f}s from {data_dir}",
        flush=True,
    )

    config = LIFConfig(seed=args.seed)
    kwargs = dict(
        config=config,
        baseline_steps=args.baseline_steps,
        stim_steps=args.stim_steps,
        recovery_steps=args.recovery_steps,
        stim_amplitude=args.stim_amplitude,
        trace_path=args.trace,
    )
    report = run_experiment(connectome, recurrent_enabled=not args.no_recurrent, **kwargs)
    report["connectome_load_seconds"] = round(load_s, 4)
    _print_report(report)

    if not args.no_recurrent:
        print("\n--- ablation: recurrent connections disabled ---\n", flush=True)
        kwargs["trace_path"] = None
        silent = run_experiment(connectome, recurrent_enabled=False, **kwargs)
        _print_report(silent)
        report["ablation_no_recurrent"] = {
            "final_spike_checksum": silent["final_spike_checksum"],
            "stimulation_phase": silent["stimulation_phase"],
            "checksum_changed": silent["final_spike_checksum"] != report["final_spike_checksum"],
            "extra_spikes_beyond_targets_with_recurrent": report["evidence"]["extra_spikes_beyond_targets"],
            "extra_spikes_beyond_targets_without_recurrent": silent["evidence"]["extra_spikes_beyond_targets"],
        }
        print(
            "\nablation: checksums differ "
            f"{report['ablation_no_recurrent']['checksum_changed']}; "
            "extra spikes beyond stimulated cells "
            f"with recurrent={report['evidence']['extra_spikes_beyond_targets']:.1f} "
            f"without={silent['evidence']['extra_spikes_beyond_targets']:.1f}",
            flush=True,
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
