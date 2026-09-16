"""Decoder, FLY CONTROL integrity, and compact-dataset tests.

Uses a tiny synthetic graph. Full MaleCNS is not required.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fly_pilot.brain.decoder import (
    DECODER_INPUT_KIND,
    CausalTemporalDecoder,
    DecoderArtifact,
    DecoderConfig,
)
from fly_pilot.brain.features import DECODER_INPUT_KIND as FEATURE_INPUT_KIND
from fly_pilot.brain.vision.scene import empty_atlas
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.trained import TrainedMaleCNSController
from fly_pilot.diagnose_decoder import dataset_diagnostics
from fly_pilot.disturbance import ControlDisturbance
from fly_pilot.record_decoder import DECODER_FORBIDDEN_INPUTS, DECODER_INPUT_COLUMNS, record_decoder_episodes
from fly_pilot.record_dagger import record_dagger_episodes
from fly_pilot.record_observing import synthetic_observer
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import AircraftControls
from fly_pilot.train_decoder import load_compact_table, load_compact_tables, split_episode_ids
from fly_pilot.verify_artifact import verify_decoder_artifact


def test_decoder_input_kind_is_dn_rates_only() -> None:
    assert DECODER_INPUT_KIND == "dn_windowed_rates"
    assert FEATURE_INPUT_KIND == "dn_windowed_rates"
    src = inspect.getsource(CausalTemporalDecoder.forward)
    for banned in ("airspeed", "along_m", "alt_agl", "heading", "roll_deg", "pitch"):
        assert banned not in src
    feat_src = inspect.getsource(TrainedMaleCNSController._decoder_features)
    assert "decoder_feature_vector" in feat_src
    assert "observe" not in feat_src or "self._obs" not in feat_src


def test_trained_controller_module_has_no_expert() -> None:
    import fly_pilot.controllers.trained as trained_mod

    src = inspect.getsource(trained_mod)
    assert "from fly_pilot.controllers.expert" not in src
    assert "import ExpertLandingController" not in src
    act_src = inspect.getsource(TrainedMaleCNSController.act)
    assert "import ExpertLandingController" not in act_src
    assert "ExpertLandingController(" not in act_src
    assert trained_mod.TrainedMaleCNSController.uses_expert is False
    assert trained_mod.TrainedMaleCNSController.uses_telemetry_as_decoder_input is False
    assert trained_mod.TrainedMaleCNSController.biological_learning is False


def test_gru_state_resets() -> None:
    torch.manual_seed(0)
    model = CausalTemporalDecoder(DecoderConfig(n_descending=6, windows=(5, 13)))
    x = np.linspace(0, 1, model.config.input_dim, dtype=np.float32)
    first = model.step_numpy(x)
    second = model.step_numpy(x)
    assert model._hidden is not None
    assert not np.allclose(first, second, atol=1e-6)
    hidden_before = model._hidden.clone()
    model.reset_state()
    assert model._hidden is None
    again = model.step_numpy(x)
    np.testing.assert_allclose(again, first, atol=1e-6)
    model.step_numpy(x)
    assert hidden_before.shape == model._hidden.shape  # type: ignore[union-attr]


def test_zero_dn_activity_changes_decoder_output() -> None:
    torch.manual_seed(1)
    model = CausalTemporalDecoder(DecoderConfig(n_descending=8))
    zeros = np.zeros(model.config.input_dim, dtype=np.float32)
    ones = np.ones(model.config.input_dim, dtype=np.float32)
    model.reset_state()
    z = model.step_numpy(zeros)
    model.reset_state()
    o = model.step_numpy(ones)
    assert z.shape == (4,)
    assert not np.allclose(z, o, atol=1e-5)


def test_dn_body_id_ordering_must_match_artifact() -> None:
    observer = synthetic_observer(n_left=3, n_right=3, n_extra=8, seed=4)
    ctl = TrainedMaleCNSController.untrained(observer, seed=4)
    ctl.artifact.assert_dn_ordering(observer.dn.spec.body_ids)
    with pytest.raises(ValueError, match="body-id"):
        ctl.artifact.assert_dn_ordering(observer.dn.spec.body_ids[::-1])


def test_decoder_outputs_are_legal() -> None:
    observer = synthetic_observer(seed=9)
    ctl = TrainedMaleCNSController.untrained(observer, seed=9)
    atlas = empty_atlas()
    observer.reset(seed=9)
    for i in range(12):
        observer.step_atlas(atlas, sim_time_s=i * 0.02, neural_step=i)
        out = ctl.act()
        assert -1.0 <= out.aileron <= 1.0
        assert -1.0 <= out.elevator <= 1.0
        assert -1.0 <= out.rudder <= 1.0
        assert 0.0 <= out.throttle <= 1.0


def test_retinal_change_can_alter_decoder_output() -> None:
    from scipy import sparse

    from fly_pilot.brain.config import LIFConfig
    from fly_pilot.brain.connectome import Connectome
    from fly_pilot.brain.observing import ObservingMaleCNS
    from fly_pilot.brain.vision.mapping import PhotoreceptorMap

    n_r, n_dn = 12, 8
    n = n_r + n_dn
    body_ids = np.arange(100, 100 + n, dtype=np.int64)
    types = np.array(["R1-R6"] * n_r + ["DNx"] * n_dn, dtype=object)
    superclasses = np.array(["ol_sensory"] * n_r + ["descending_neuron"] * n_dn, dtype=object)
    sides = np.array(["L"] * (n_r // 2) + ["R"] * (n_r - n_r // 2) + ["L"] * (n_dn // 2) + ["R"] * (n_dn - n_dn // 2), dtype=object)
    pre = np.repeat(np.arange(n_r, dtype=np.int32), n_dn)
    post = np.tile(np.arange(n_r, n, dtype=np.int32), n_r)
    w = np.full(pre.size, 4.0, dtype=np.float32)
    counts = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    connectome = Connectome.from_parts(
        body_ids,
        counts,
        type=types,
        flywire_type=types,
        instance=np.array([f"{t}_{s}" for t, s in zip(types, sides)], dtype=object),
        superclass=superclasses,
        side=sides,
        consensus_nt=np.array(["acetylcholine"] * n, dtype=object),
        sign=np.ones(n, dtype=np.int8),
    )
    mapping = PhotoreceptorMap.synthetic(n_r // 2, n_r - n_r // 2, start_index=0)
    mapping.indices = np.arange(n_r, dtype=np.int32)
    mapping.body_ids = body_ids[:n_r]
    observer = ObservingMaleCNS(
        connectome,
            config=LIFConfig(seed=11, tonic_current=0.05, background_rate_hz=0.0, synaptic_gain=2.0),
        mapping=mapping,
    )
    ctl = TrainedMaleCNSController.untrained(observer, seed=11)
    dark = empty_atlas()
    bright = np.full_like(dark, 220)

    def _roll(atlas: np.ndarray) -> tuple[AircraftControls, np.ndarray]:
        ctl.reset()
        observer.reset(seed=11)
        last = AircraftControls()
        for i in range(20):
            observer.step_atlas(atlas, sim_time_s=i * 0.02, neural_step=i)
            last = ctl.act()
        return last, observer.dn.decoder_feature_vector().copy()

    a, dark_dn = _roll(dark)
    b, bright_dn = _roll(bright)
    assert float(np.linalg.norm(dark_dn - bright_dn)) > 1e-6
    delta = (
        abs(a.aileron - b.aileron)
        + abs(a.elevator - b.elevator)
        + abs(a.rudder - b.rudder)
        + abs(a.throttle - b.throttle)
    )
    assert delta > 1e-4


def test_fly_control_sandbox_does_not_use_expert() -> None:
    observer = synthetic_observer(seed=12)
    ctl = TrainedMaleCNSController.untrained(observer, seed=12)
    sandbox = LandingSandbox(controller=ctl, observer=observer, randomize_spawns=True)
    sandbox.reset(seed=3)
    assert sandbox.mode == "fly_control"
    assert sandbox.controller.name == "fly_control"
    assert not isinstance(sandbox.controller, ExpertLandingController)
    expert = ExpertLandingController()
    for _ in range(40):
        obs = sandbox.aircraft.observe()
        expert.observe(obs)
        expert_cmd = expert.act()
        snap = sandbox.step_once()
        applied = sandbox.last_applied_controls
        assert applied is not None
        assert not hasattr(sandbox.controller, "phase")
        # Decoder is untrained; it must still be the authority even if commands coincide.
        assert sandbox.controller.name == "fly_control"
        if snap.episode.status.value != "in_progress":
            break
    assert sandbox.mode == "fly_control"
    _ = expert_cmd


def test_set_controller_rejects_untrained_malecns_name() -> None:
    sandbox = LandingSandbox()
    with pytest.raises(ValueError, match="Untrained MaleCNSController"):
        sandbox.set_controller("malecns")


def test_compact_dataset_has_only_dn_and_expert_actions(tmp_path: Path) -> None:
    out = tmp_path / "dec.parquet"
    meta = record_decoder_episodes(out, successes=1, seed=21, synthetic=True, max_steps=120, disturb=True)
    assert meta["contains_aircraft_telemetry"] is False
    assert meta["contains_retinal_blobs"] is False
    assert meta["decoder_input"] == list(DECODER_INPUT_COLUMNS)
    import pyarrow.parquet as pq

    table = pq.read_table(out)
    cols = set(table.column_names)
    for required in (
        "episode_id",
        "timestep",
        "sim_time_s",
        "dn_rates_100_f16",
        "dn_rates_260_f16",
        "expert_aileron",
        "expert_elevator",
        "expert_rudder",
        "expert_throttle",
    ):
        assert required in cols
    for banned in DECODER_FORBIDDEN_INPUTS:
        assert banned not in cols
    assert "retinal_currents_f16" not in cols
    bundle = load_compact_table(out)
    assert bundle["dn_n"] > 0
    split = split_episode_ids(sorted(bundle["episodes"]), train=1, val=1, test=1, seed=0)
    assert "train" in split
    # Episode-level split uses whole ids, not rows.
    assert all(isinstance(i, int) for i in split["train"])


def test_disturbance_changes_applied_not_expert_target() -> None:
    dist = ControlDisturbance.plan(0, enabled=True)
    assert dist.pulses
    expert = AircraftControls(aileron=0.1, elevator=-0.1, rudder=0.0, throttle=0.4)
    pulse = dist.pulses[0]
    applied = dist.apply(expert, sim_time_s=(pulse.start_s + pulse.end_s) / 2, alt_agl_m=120.0)
    assert applied.aileron != expert.aileron or applied.elevator != expert.elevator
    high = dist.apply(expert, sim_time_s=(pulse.start_s + pulse.end_s) / 2, alt_agl_m=10.0)
    assert high.aileron == pytest.approx(expert.aileron)


def test_train_tiny_decoder_on_synthetic_episodes(tmp_path: Path) -> None:
    data = tmp_path / "tiny.parquet"
    record_decoder_episodes(data, successes=3, seed=30, synthetic=True, max_steps=200, disturb=True)
    ckpt = tmp_path / "best.pt"
    from fly_pilot.train_decoder import train_decoder

    report = train_decoder(
        data,
        ckpt,
        seed=0,
        epochs=2,
        patience=2,
        batch_size=4,
        seq_len=8,
        stride=4,
        train_episodes=2,
        val_episodes=1,
        test_episodes=1,
        gru_hidden=16,
    )
    assert ckpt.exists()
    assert report["parameter_count"] > 0
    loaded = DecoderArtifact.load(ckpt)
    assert loaded.biological_learning is False
    assert loaded.config.input_kind == "dn_windowed_rates"
    observer = synthetic_observer(seed=2)
    ctl = TrainedMaleCNSController.untrained(observer, seed=2)
    path = tmp_path / "tiny.pt"
    ctl.artifact.save(path)
    loaded = DecoderArtifact.load(path)
    loaded.assert_dn_ordering(observer.dn.spec.body_ids)
    assert loaded.biological_learning is False
    assert loaded.config.input_kind == "dn_windowed_rates"
    x = observer.dn.decoder_feature_vector()
    if x.size == 0:
        x = np.zeros(loaded.config.input_dim, dtype=np.float32)
    a = loaded.model.step_numpy(x)
    assert a.shape == (4,)


def test_multiple_compact_datasets_remap_complete_episodes(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    record_decoder_episodes(first, successes=1, seed=40, synthetic=True, max_steps=80)
    record_decoder_episodes(second, successes=1, seed=41, synthetic=True, max_steps=80)
    bundle = load_compact_tables([first, second])
    assert sorted(bundle["episodes"]) == [0, 1]
    assert bundle["episode_sources"][0]["source_episode_id"] == 0
    assert bundle["episode_sources"][1]["source_episode_id"] == 0
    assert bundle["episode_sources"][0]["path"] != bundle["episode_sources"][1]["path"]
    split = split_episode_ids(sorted(bundle["episodes"]), train=1, val=1, test=1, seed=0)
    assert set(split["train"]).isdisjoint(split["test"])
    diagnostics = dataset_diagnostics([first, second])
    assert set(diagnostics["temporal_phase_features"]) == {"opening", "middle", "terminal"}
    assert diagnostics["features"]["count"] == bundle["dn_n"] * 2


def test_dagger_uses_fly_authority_and_shadow_labels_only(tmp_path: Path) -> None:
    observer = synthetic_observer(seed=51)
    controller = TrainedMaleCNSController.untrained(observer, seed=51)
    output = tmp_path / "dagger.parquet"
    report = record_dagger_episodes(
        output,
        episodes=1,
        seed=51,
        checkpoint=tmp_path / "not-needed.pt",
        observer=observer,
        controller=controller,
        max_steps=100,
    )
    assert report["control_authority"] == "fly_control"
    assert report["expert_controls_aircraft"] is False
    assert report["commands_blended"] is False
    assert report["rows"] > 0
    import pyarrow.parquet as pq

    columns = set(pq.read_schema(output).names)
    assert set(DECODER_INPUT_COLUMNS).issubset(columns)
    assert not (set(DECODER_FORBIDDEN_INPUTS) & columns)


def test_artifact_metadata_hash_and_smoke_verification(tmp_path: Path) -> None:
    observer = synthetic_observer(seed=60)
    controller = TrainedMaleCNSController.untrained(observer, seed=60)
    controller.artifact.git_commit = "abc123"
    controller.artifact.git_dirty = True
    controller.artifact.training_command = ["python", "-m", "fly_pilot.train_decoder"]
    controller.artifact.dependencies = {"python": "test"}
    controller.artifact.seed_ranges = {"test": {"start": 4000, "episodes": 100}}
    checkpoint = tmp_path / "best.pt"
    controller.artifact.save(checkpoint)
    report = verify_decoder_artifact(checkpoint, check_prepared=False)
    assert report["ok"] is True
    assert report["training_data_required"] is False
    assert report["dn_order_check"] == "embedded-structural"

    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_decoder_artifact(checkpoint, check_prepared=False)


def test_failed_fly_switch_preserves_current_controller(tmp_path: Path) -> None:
    observer = synthetic_observer(seed=70)
    sandbox = LandingSandbox(observer=observer, decoder_path=tmp_path / "missing.pt")
    original = sandbox.controller
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        sandbox.set_controller("fly_control")
    assert sandbox.controller is original
    assert sandbox.mode == "manual"


def test_fixed_matrix_ranks_success_then_progress_then_offline_mae(monkeypatch, tmp_path: Path) -> None:
    import fly_pilot.select_decoder as selection

    mae = {
        "expert-full": 0.01,
        "expert-small": 0.02,
        "dagger-full": 0.04,
        "dagger-small": 0.03,
    }

    def fake_train(_data, checkpoint, **_kwargs):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        for name in ("best.meta.json", "train_report.json", "split.json"):
            (checkpoint.parent / name).write_text("{}\n")
        value = mae[checkpoint.parent.name]
        controls = {name: {"mae": value} for name in ("aileron", "elevator", "rudder", "throttle")}
        return {"offline": {"test": {"gru": controls}}}

    def fake_evaluate(_episodes, _seed, *, checkpoint, json_out, **_kwargs):
        name = checkpoint.parent.name
        success = {"expert-full": 0, "expert-small": 1, "dagger-full": 2, "dagger-small": 2}[name]
        row = SimpleNamespace(
            centerline_error_m=10.0,
            heading_error_deg=2.0,
            along_m=-100.0,
            success=False,
        )
        payload = {"summary": {"success_count": success, "success_rate": success / 20.0}}
        json_out.write_text(json.dumps(payload))
        return [row], payload

    monkeypatch.setattr(selection, "train_decoder", fake_train)
    monkeypatch.setattr(selection, "evaluate_fly", fake_evaluate)
    final = tmp_path / "final"
    report = selection.run_matrix(
        tmp_path / "expert.parquet",
        [tmp_path / "d1.parquet", tmp_path / "d2.parquet"],
        tmp_path / "candidates",
        final_dir=final,
    )
    assert report["winner"]["name"] == "dagger-small"
    assert report["ranking"][:2] == ["dagger-small", "dagger-full"]
    assert (final / "best.pt").read_bytes() == b"checkpoint"
    assert (final / "model-selection.json").exists()
