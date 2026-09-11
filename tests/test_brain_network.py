"""Full-network smoke tests. Skipped unless MaleCNS data has been prepared."""

from __future__ import annotations

from pathlib import Path

import pytest

from fly_pilot.brain.config import EXPECTED_EDGE_COUNT, EXPECTED_NEURON_COUNT, LIFConfig, default_data_dir, prepared_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.demo import run_experiment
from fly_pilot.brain.model import MaleCNSLIF
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.replay import SpikeTrace


def _prepared() -> Path | None:
    dest = prepared_dir(default_data_dir())
    if (dest / "manifest.json").exists() and (dest / "weights.npz").exists():
        return dest
    return None


pytestmark = pytest.mark.skipif(_prepared() is None, reason="run python -m fly_pilot.brain.prepare")


def test_full_connectome_counts() -> None:
    connectome = Connectome.load()
    assert connectome.n_neurons == EXPECTED_NEURON_COUNT
    assert connectome.n_edges == EXPECTED_EDGE_COUNT
    assert connectome.synapse_count_sum > connectome.n_edges
    assert connectome.index_of(int(connectome.body_ids[0])) == 0


def test_full_network_one_step_and_population_query() -> None:
    connectome = Connectome.load()
    index = NeuronIndex(connectome)
    assert index.count(superclass="descending_neuron") >= 1000
    assert index.count(cell_type="R1-R6") >= 1000
    assert index.count(cell_type="DNp01") == 2
    net = MaleCNSLIF(connectome, LIFConfig(seed=64))
    result = net.step()
    assert result.fired.shape == (connectome.n_neurons,)
    assert 0 <= result.n_spikes <= connectome.n_neurons


def test_full_network_replay_and_stimulation_ablation() -> None:
    connectome = Connectome.load()
    cfg = LIFConfig(seed=64)
    kwargs = dict(
        config=cfg,
        baseline_steps=4,
        stim_steps=4,
        recovery_steps=4,
        stim_amplitude=1.2,
        trace_path=None,
    )
    with_rec = run_experiment(connectome, recurrent_enabled=True, **kwargs)
    no_rec = run_experiment(connectome, recurrent_enabled=False, **kwargs)
    assert with_rec["evidence"]["stim_entered_R1R6"]
    assert with_rec["final_spike_checksum"] != no_rec["final_spike_checksum"]
    assert with_rec["stimulation_phase"]["total_spikes"] != no_rec["stimulation_phase"]["total_spikes"]
    assert with_rec["stimulation_phase"]["mean_R1-R6_hz"] > 20.0

    again = run_experiment(connectome, recurrent_enabled=True, **kwargs)
    assert again["final_spike_checksum"] == with_rec["final_spike_checksum"]
    trace = SpikeTrace(
        seed=64,
        n_neurons=connectome.n_neurons,
        n_steps=12,
        dt=cfg.dt,
        stimulation={},
        step_checksums=[],
        step_spike_counts=[],
        final_checksum=with_rec["final_spike_checksum"],
    )
    assert trace.final_checksum == again["final_spike_checksum"]
