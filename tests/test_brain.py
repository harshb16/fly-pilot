"""Unit tests for the standalone MaleCNS LIF module (synthetic graphs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse

from fly_pilot.brain.config import INHIBITORY_TRANSMITTERS, LIFConfig
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.data import (
    aggregate_edges,
    build_from_tables,
    map_body_ids,
    transmitter_sign,
)
from fly_pilot.brain.model import MaleCNSLIF
from fly_pilot.brain.populations import NeuronIndex
from fly_pilot.brain.replay import SpikeTrace, traces_match
from fly_pilot.brain.stimulation import PulseStimulation


def _tiny_connectome() -> Connectome:
    """Three neurons: 0 excitatory → 1, 2 inhibitory → 1."""
    body_ids = np.array([10, 20, 30], dtype=np.int64)
    # counts[post, pre]
    counts = sparse.csr_matrix(
        ([4.0, 5.0], ([1, 1], [0, 2])),
        shape=(3, 3),
        dtype=np.float32,
    )
    return Connectome.from_parts(
        body_ids,
        counts,
        type=np.array(["R1-R6", "L1", "DNp01"], dtype=object),
        flywire_type=np.array(["R1-6", "L1", "DNp01"], dtype=object),
        instance=np.array(["R1-R6_R", "L1_L", "DNp01(GF)_R"], dtype=object),
        superclass=np.array(["ol_sensory", "ol_intrinsic", "descending_neuron"], dtype=object),
        class_name=np.array(["", "", ""], dtype=object),
        subclass=np.array(["", "", "lt"], dtype=object),
        side=np.array(["R", "L", "R"], dtype=object),
        consensus_nt=np.array(["acetylcholine", "acetylcholine", "gaba"], dtype=object),
        sign=np.array([1, 1, -1], dtype=np.int8),
    )


def test_map_body_ids_dense_index() -> None:
    ids = np.array([10, 20, 30], dtype=np.int64)
    index, valid = map_body_ids(ids, np.array([30, 10, 99, 20], dtype=np.int64))
    assert valid.tolist() == [True, True, False, True]
    assert index[valid].tolist() == [2, 0, 1]


def test_edge_aggregation_sums_duplicate_pairs() -> None:
    pre = np.array([0, 0, 1], dtype=np.int32)
    post = np.array([1, 1, 2], dtype=np.int32)
    weight = np.array([3, 2, 7], dtype=np.float32)
    matrix = aggregate_edges(pre, post, weight, 3)
    assert matrix.nnz == 2
    assert matrix[1, 0] == 5
    assert matrix[2, 1] == 7


def test_transmitter_sign_gaba_glutamate_histamine_inhibitory() -> None:
    labels = np.array(
        ["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "unclear", "GABA", ""],
        dtype=object,
    )
    sign = transmitter_sign(labels)
    assert sign.tolist() == [1, -1, -1, -1, 1, 1, -1, 1]
    assert INHIBITORY_TRANSMITTERS == frozenset({"gaba", "glutamate", "histamine"})


def test_prepare_tables_drops_empty_superclass_and_aggregates(tmp_path: Path) -> None:
    annotations = pa.table(
        {
            "bodyId": [10, 20, 30, 40],
            "superclass": ["ol_sensory", "ol_intrinsic", "descending_neuron", ""],
            "type": ["R1-R6", "L1", "DNp01", "junk"],
            "flywireType": ["R1-6", "L1", "DNp01", ""],
            "instance": ["R1-R6_R", "L1_L", "DNp01(GF)_R", "x"],
            "class": ["", "", "", ""],
            "subclass": ["", "", "lt", ""],
            "somaSide": ["R", "L", "R", ""],
            "rootSide": ["", "", "", ""],
        }
    )
    transmitters = pa.table(
        {
            "body": [10, 10, 20, 30],
            "consensus_nt": ["acetylcholine", "should_be_ignored", "glutamate", "gaba"],
        }
    )
    weights = pa.RecordBatch.from_pydict(
        {
            "body_pre": [10, 10, 20, 10, 99],
            "body_post": [20, 20, 30, 10, 20],
            "weight": [4, 1, 2, 3, 50],
        }
    )
    ids, meta, matrix, stats = build_from_tables(
        annotations,
        transmitters,
        [weights],
        expected_neurons=None,
        expected_edges=None,
    )
    assert ids.tolist() == [10, 20, 30]
    assert matrix[1, 0] == 5  # 4+1 aggregated
    assert matrix[2, 1] == 2
    assert matrix[0, 0] == 3  # autapse kept
    assert 99 not in ids
    assert stats["mapped_weight_rows"] == 4  # dropped the fragment endpoint
    assert stats["n_edges"] == 3
    signs = np.asarray(meta["sign"].to_numpy())
    nts = meta["consensus_nt"].to_pylist()
    assert nts == ["acetylcholine", "glutamate", "gaba"]
    assert signs.tolist() == [1, -1, -1]


def test_prepare_writes_compact_artifacts() -> None:
    connectome = _tiny_connectome()
    assert connectome.n_neurons == 3
    assert connectome.n_edges == 2
    assert connectome.index_of(20) == 1
    assert connectome.synapse_count_sum == 9


def test_signed_weights_flip_inhibitory_presynaptic_cells() -> None:
    connectome = _tiny_connectome()
    weights = connectome.signed_normalized_weights(LIFConfig(normalize_incoming=False))
    assert weights[1, 0] > 0
    assert weights[1, 2] < 0


def test_incoming_normalization_is_per_postsynaptic_row() -> None:
    connectome = _tiny_connectome()
    weights = connectome.signed_normalized_weights(LIFConfig(normalize_incoming=True))
    # |4| + |5| = 9 onto neuron 1
    assert weights[1, 0] == pytest.approx(4 / 9)
    assert weights[1, 2] == pytest.approx(-5 / 9)


def _silent_config(**kwargs) -> LIFConfig:
    defaults = dict(
        dt=0.02,
        tau_m=0.1,
        v_threshold=1.0,
        v_reset=0.0,
        tonic_current=0.0,
        synaptic_gain=1.0,
        background_rate_hz=0.0,
        background_amplitude=0.0,
        normalize_incoming=False,
        seed=0,
    )
    defaults.update(kwargs)
    return LIFConfig(**defaults)


def test_lif_timestep_spikes_and_resets() -> None:
    connectome = _tiny_connectome()
    net = MaleCNSLIF(connectome, _silent_config(), recurrent_enabled=False)
    net.voltage[0] = 0.4
    stim = PulseStimulation(indices=np.array([0]), start_step=0, end_step=1, amplitude=0.7)
    result = net.step(stim)
    assert result.fired[0]
    assert not result.fired[1]
    assert net.voltage[0] == 0.0  # reset
    assert net.spikes[0] == 1.0


def test_reset_clears_voltage_and_reseeds() -> None:
    connectome = _tiny_connectome()
    net = MaleCNSLIF(connectome, _silent_config(background_rate_hz=20.0, background_amplitude=0.5, seed=1))
    net.voltage[:] = 0.9
    net.step()
    net.reset()
    assert np.all(net.voltage == 0)
    assert net.step_count == 0
    first = net.step().checksum
    net.reset()
    second = net.step().checksum
    assert first == second


def test_seeded_background_is_deterministic_and_seed_dependent() -> None:
    connectome = _tiny_connectome()
    cfg_a = _silent_config(background_rate_hz=20.0, background_amplitude=1.2, seed=7)
    cfg_b = _silent_config(background_rate_hz=20.0, background_amplitude=1.2, seed=8)
    a1 = MaleCNSLIF(connectome, cfg_a)
    a2 = MaleCNSLIF(connectome, cfg_a)
    b = MaleCNSLIF(connectome, cfg_b)
    checks_a1 = [a1.step().checksum for _ in range(8)]
    checks_a2 = [a2.step().checksum for _ in range(8)]
    checks_b = [b.step().checksum for _ in range(8)]
    assert checks_a1 == checks_a2
    assert checks_a1 != checks_b


def test_population_lookup_by_annotations() -> None:
    index = NeuronIndex(_tiny_connectome())
    assert index.count(superclass="descending_neuron") == 1
    assert index.count(cell_type="R1-6") == 1  # alias of R1-R6
    assert index.count(type="L1", side="L") == 1
    assert index.count(inhibitory=True) == 1
    assert index.body_ids(cell_type="DNp01")[0] == 30


def test_stimulation_enters_target_neurons_only() -> None:
    connectome = _tiny_connectome()
    net = MaleCNSLIF(connectome, _silent_config(), recurrent_enabled=False)
    stim = PulseStimulation(indices=np.array([2]), start_step=0, end_step=1, amplitude=1.5)
    result = net.step(stim)
    assert result.n_stimulated == 1
    assert result.fired[2]
    assert not result.fired[0]
    assert not result.fired[1]


def test_recurrent_sign_propagates_to_postsynaptic_voltage() -> None:
    connectome = _tiny_connectome()
    # Force neuron 0 to spike, no tonic/noise, watch neuron 1.
    net = MaleCNSLIF(connectome, _silent_config(synaptic_gain=1.0), recurrent_enabled=True)
    net.spikes[0] = 1.0
    net.step()  # applies W @ previous spikes into voltage; may or may not spike
    # Incoming from 0 is +4 (unsigned), so v[1] should be positive.
    # After the step, if v[1] did not reach threshold it remains the leaked+input value.
    # Reconstruct one step more carefully from a reset:
    net.reset()
    net.spikes[:] = 0
    net.spikes[0] = 1.0
    before = net.voltage.copy()
    net.step()
    assert net.last_recurrent[1] == pytest.approx(4.0)
    assert net.last_recurrent[1] > before[1]

    net.reset()
    net.spikes[:] = 0
    net.spikes[2] = 1.0  # inhibitory onto 1
    net.step()
    assert net.last_recurrent[1] == pytest.approx(-5.0)


def test_disabling_recurrent_zeros_anatomical_input() -> None:
    connectome = _tiny_connectome()
    silent = _silent_config()
    with_rec = MaleCNSLIF(connectome, silent, recurrent_enabled=True)
    no_rec = MaleCNSLIF(connectome, silent, recurrent_enabled=False)
    with_rec.spikes[0] = 1.0
    no_rec.spikes[0] = 1.0
    with_rec.step()
    no_rec.step()
    assert with_rec.last_recurrent[1] != 0
    assert no_rec.last_recurrent[1] == 0


def test_deterministic_replay_checksums_match() -> None:
    connectome = _tiny_connectome()
    cfg = _silent_config(background_rate_hz=10.0, background_amplitude=0.4, seed=64)
    stim = PulseStimulation(indices=np.array([0, 2]), start_step=2, end_step=5, amplitude=1.1)

    def run() -> SpikeTrace:
        net = MaleCNSLIF(connectome, cfg)
        trace = SpikeTrace(
            seed=cfg.seed,
            n_neurons=connectome.n_neurons,
            n_steps=8,
            dt=cfg.dt,
            stimulation={"start": 2, "end": 5},
        )
        for _ in range(8):
            trace.record(net.step(stim).fired)
        return trace

    assert traces_match(run(), run())


def test_replay_trace_roundtrip(tmp_path: Path) -> None:
    connectome = _tiny_connectome()
    cfg = _silent_config(seed=3, background_rate_hz=5.0, background_amplitude=0.2)
    net = MaleCNSLIF(connectome, cfg)
    trace = SpikeTrace(seed=3, n_neurons=3, n_steps=4, dt=0.02, stimulation={})
    for _ in range(4):
        trace.record(net.step().fired)
    path = tmp_path / "trace.json"
    trace.write(path)
    loaded = SpikeTrace.read(path)
    assert traces_match(trace, loaded)
    net.reset()
    rerun = SpikeTrace(seed=3, n_neurons=3, n_steps=4, dt=0.02, stimulation={})
    for _ in range(4):
        rerun.record(net.step().fired)
    assert traces_match(loaded, rerun)
