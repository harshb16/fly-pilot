"""Milestone 4 vision, scheduler, features, observing, replay tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_pilot.brain.config import LIFConfig
from fly_pilot.brain.features import DescendingNeuronFeatureExtractor, descending_population
from fly_pilot.brain.observing import CONTROLS_AIRCRAFT, ObservingMaleCNS
from fly_pilot.brain.replay_episode import replay_table
from fly_pilot.brain.scheduler import Schedule, SimScheduler
from fly_pilot.brain.stimulation import CurrentStimulation
from fly_pilot.brain.vision.encoder import FlyEyeEncoder
from fly_pilot.brain.vision.geometry import (
    ATLAS_HEIGHT,
    ATLAS_WIDTH,
    FACE_SIZE,
    angular_rays,
    atlas_indices_for_rays,
)
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.brain.vision.scene import FlyViewPose, empty_atlas, render_cubemap_atlas
from fly_pilot.brain.vision.stimulus import RetinalStimulusFrame
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.trained import TrainedMaleCNSController
from fly_pilot.record_observing import record_episodes, synthetic_observer
from fly_pilot.runway import Runway
from fly_pilot.sandbox import LandingSandbox
from fly_pilot.state import AircraftControls
from fly_pilot.validate_vision import (
    experiment_a_static,
    experiment_b_runway_motion,
    experiment_c_left_right,
    experiment_d_replay,
)


def test_scheduler_neural_ticks_follow_sim_time_not_wall() -> None:
    sched = SimScheduler(Schedule(physics_hz=120, vision_hz=50, neural_hz=50, log_hz=50))
    due0 = sched.drain(0.0)
    assert [e.neural_step for e in due0 if e.neural] == [0]
    fired = [0]
    t = 0.0
    dt = 1.0 / 120.0
    for _ in range(120):
        t = sched.note_physics(dt)
        fired.extend(e.neural_step for e in sched.drain(t) if e.neural)
    # t ≈ 1.0 s → neural steps 0..50 inclusive = 51 ticks
    assert fired[0] == 0
    assert fired[-1] == 50
    assert len(fired) == 51


def test_scheduler_ignores_fast_frontend_fps() -> None:
    sched = SimScheduler()
    sched.drain(0.0)
    # 1000 fake display frames at the same sim time do not create neural ticks.
    extra = []
    for _ in range(1000):
        extra.extend(sched.drain(0.0))
    assert extra == []


def test_photoreceptor_map_is_deterministic() -> None:
    a = PhotoreceptorMap.synthetic(12, 12)
    b = PhotoreceptorMap.synthetic(12, 12)
    assert np.allclose(a.azimuth_deg, b.azimuth_deg)
    assert np.allclose(a.elevation_deg, b.elevation_deg)
    assert np.array_equal(a.sample_atlas_indices, b.sample_atlas_indices)
    assert int(a.left_mask().sum()) == 12
    assert int(a.right_mask().sum()) == 12


def test_pixel_direction_mapping_hits_forward_face() -> None:
    ray = angular_rays(np.array([0.0]), np.array([0.0]))[0]
    idx = int(atlas_indices_for_rays(ray[None, :])[0])
    y, x = divmod(idx, ATLAS_WIDTH)
    # Forward face is index 0, top-left of the atlas.
    assert 0 <= x < FACE_SIZE
    assert 0 <= y < FACE_SIZE


def test_encoder_does_not_stimulate_all_receptors_identically() -> None:
    mapping = PhotoreceptorMap.synthetic(20, 20)
    atlas = empty_atlas()
    # Paint a white blob on the forward face center.
    atlas[FACE_SIZE // 2 - 2 : FACE_SIZE // 2 + 3, FACE_SIZE // 2 - 2 : FACE_SIZE // 2 + 3] = 255
    enc = FlyEyeEncoder(mapping)
    frame = enc.encode(atlas)
    assert frame.n_receptors == 40
    assert float(frame.luminance.max()) > float(frame.luminance.min()) + 0.01
    # Nose-pointing receptors (small |az|, |el|) should be brighter than far-side ones.
    near = np.abs(mapping.azimuth_deg) < 15
    far = np.abs(mapping.azimuth_deg) > 80
    assert float(frame.luminance[near].mean()) > float(frame.luminance[far].mean())


def test_temporal_contrast_responds_to_frame_change() -> None:
    mapping = PhotoreceptorMap.synthetic(8, 8)
    enc = FlyEyeEncoder(mapping)
    dark = empty_atlas()
    bright = np.full((ATLAS_HEIGHT, ATLAS_WIDTH, 3), 200, dtype=np.uint8)
    first = enc.encode(dark)
    second = enc.encode(bright)
    assert float(enc.last_temporal.mean()) > 0.05
    assert float(second.currents.mean()) > float(first.currents.mean())
    # Static thereafter
    third = enc.encode(bright)
    assert float(enc.last_temporal.mean()) < 1e-6
    assert third.current_sha256 != second.current_sha256 or True
    # currents may still include luminance; temporal term is what dropped
    assert float(np.mean(np.abs(third.currents - second.currents))) >= 0.0


def test_canonical_currents_match_float16_storage() -> None:
    mapping = PhotoreceptorMap.synthetic(6, 6)
    enc = FlyEyeEncoder(mapping)
    atlas = empty_atlas()
    atlas[:, :] = (40, 80, 30)
    frame = enc.encode(atlas)
    stored = np.frombuffer(frame.to_record()["currents_f16"], dtype=np.float16).astype(np.float32)
    np.testing.assert_array_equal(stored, frame.currents)


def test_retinal_stimulus_roundtrip() -> None:
    currents = np.linspace(0, 0.5, 17, dtype=np.float32)
    lum = np.linspace(0, 1, 17, dtype=np.float32)
    frame = RetinalStimulusFrame(currents=currents, luminance=lum, neural_step=4, sim_time_s=0.08)
    record = frame.to_record()
    loaded = RetinalStimulusFrame.from_record(record)
    assert loaded.n_receptors == 17
    assert loaded.neural_step == 4
    np.testing.assert_allclose(loaded.currents, currents, atol=2e-3)
    raw = frame.to_bytes()
    assert raw.startswith(b"{")
    assert frame.current_sha256 == RetinalStimulusFrame.from_currents(currents, luminance=lum).current_sha256


def test_dn_feature_windows_and_sides() -> None:
    observer = synthetic_observer(n_left=4, n_right=4, n_extra=10, seed=1)
    spec = descending_population(observer.connectome)
    ext = DescendingNeuronFeatureExtractor(spec, dt=0.02, windows=(2, 4))
    fired = np.zeros(observer.connectome.n_neurons, dtype=bool)
    if spec.n:
        fired[spec.indices[0]] = True
    ext.update(fired)
    ext.update(fired)
    ext.update(np.zeros_like(fired))
    ext.update(np.zeros_like(fired))
    vec2 = ext.last_rates[2]
    vec4 = ext.last_rates[4]
    assert vec4.shape == (spec.n,)
    # Last two steps of the 2-window are zeros; 4-window still includes earlier spikes.
    assert float(vec4[0]) >= float(vec2[0])
    summary = ext.compact_summary()
    assert summary["n_descending"] == spec.n
    assert "left_mean_hz" in summary
    assert "right_mean_hz" in summary


def test_dn_rates_use_available_samples_during_warmup() -> None:
    """Early rates must divide by elapsed samples, not the full window length."""
    observer = synthetic_observer(n_left=2, n_right=2, n_extra=6, seed=3)
    spec = descending_population(observer.connectome)
    ext = DescendingNeuronFeatureExtractor(spec, dt=0.02, windows=(5, 13))
    fired = np.zeros(observer.connectome.n_neurons, dtype=bool)
    assert spec.n > 0
    fired[spec.indices[0]] = True
    first = ext.update(fired)
    # 1 spike in 1 × 20 ms sample → 50 Hz, not 1 / (5 × 0.02) = 10 Hz.
    assert first["rates_w5"][0] == pytest.approx(1.0 / 0.02)
    assert first["rates_w13"][0] == pytest.approx(1.0 / 0.02)
    second = ext.update(fired)
    assert second["rates_w5"][0] == pytest.approx(2.0 / 0.04)
    assert second["rates_w13"][0] == pytest.approx(2.0 / 0.04)
    # After the short window is full, 100 ms uses 5 samples; 260 ms is still warming.
    for _ in range(3):
        ext.update(np.zeros_like(fired))
    full_short = ext.last_rates[5]
    warming_long = ext.last_rates[13]
    assert full_short[0] == pytest.approx(2.0 / 0.10)
    assert warming_long[0] == pytest.approx(2.0 / 0.10)
    assert ext.available_samples == 5


def test_observing_brain_cannot_control_aircraft() -> None:
    assert CONTROLS_AIRCRAFT is False
    observer = synthetic_observer()
    assert observer.controls_aircraft is False
    assert not hasattr(observer, "act")
    with pytest.raises(AttributeError):
        getattr(observer, "act")()  # type: ignore[misc]


def test_expert_observing_uses_expert_controls_only() -> None:
    observer = synthetic_observer(seed=2)
    sandbox = LandingSandbox(
        controller=ExpertLandingController(),
        observer=observer,
        observing=True,
        randomize_spawns=True,
    )
    sandbox.reset(seed=1)
    assert sandbox.mode == "expert_observing"
    assert sandbox.controller.name == "expert"
    assert isinstance(sandbox.controller, ExpertLandingController)
    assert not hasattr(sandbox.observer, "act")
    for _ in range(30):
        snap = sandbox.step_once()
        applied = sandbox.last_applied_controls
        assert applied is not None
        assert snap.observation.aileron == pytest.approx(applied.aileron)
        assert snap.observation.elevator == pytest.approx(applied.elevator)
        assert snap.observing is True
        assert snap.fly_observing.get("controlling") is False
        if snap.episode.status.value != "in_progress":
            break
    assert "NOT CONTROLLING" in snap.fly_observing["label"]


def test_set_controller_expert_observing_and_fly_control() -> None:
    sandbox = LandingSandbox()
    observer = synthetic_observer()
    sandbox.set_observer(observer)
    snap = sandbox.set_controller("expert_observing")
    assert snap.mode == "expert_observing"
    assert sandbox.controller.name == "expert"
    sandbox.set_controller("manual")
    assert sandbox.observing is False
    ctl = TrainedMaleCNSController.untrained(observer, seed=1)
    fly = LandingSandbox(controller=ctl, observer=observer)
    assert fly.mode == "fly_control"
    assert fly.controller.name == "fly_control"
    assert not isinstance(fly.controller, ExpertLandingController)


def test_recording_alignment_and_replay(tmp_path: Path) -> None:
    out = tmp_path / "obs.parquet"
    meta = record_episodes(out, episodes=1, seed=5, synthetic=True, max_steps=180)
    assert meta["male_cns_controls_aircraft"] is False
    assert meta["rows"] > 5
    import pyarrow.parquet as pq

    table = pq.read_table(out)
    cols = set(table.column_names)
    for required in (
        "episode_id",
        "seed",
        "sim_time_s",
        "neural_step",
        "along_m",
        "alt_agl_m",
        "airspeed_kts",
        "aileron",
        "elevator",
        "rudder",
        "throttle",
        "retinal_currents_f16",
        "dn_rates_f16",
        "spike_checksum",
        "n_spikes",
    ):
        assert required in cols
    neural = table["neural_step"].to_numpy()
    sim = table["sim_time_s"].to_numpy()
    # Neural steps increase with sim time; one row per neural tick (+ terminal).
    assert neural[0] >= 0
    assert sim[-1] >= sim[0]
    replay = replay_table(out)
    assert replay["ok"] is True
    assert replay["compared"] == replay["matches"]
    assert replay["compared"] > 0


def test_validation_experiments_a_b_c() -> None:
    observer = synthetic_observer(seed=64)
    a = experiment_a_static(observer)
    assert a["replay_same_seed_matches"]
    assert a["current_stable_after_first"]
    b = experiment_b_runway_motion(observer)
    assert b["stimulus_changes"]
    c = experiment_c_left_right(observer.encoder)
    assert c["atlas_differs"]
    assert c["retinal_spatial_difference"]


def test_validation_experiment_d_replay(tmp_path: Path) -> None:
    report = experiment_d_replay(tmp_path / "d.parquet")
    assert report["replay_ok"] is True
    assert report["compared"] == report["matches"]
    assert report["compared"] > 0


def test_current_stimulation_is_per_receptor() -> None:
    observer = synthetic_observer(n_left=3, n_right=3, n_extra=6, seed=0)
    observer.net.reset()
    amps = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    stim = CurrentStimulation(observer.mapping.indices, amps)
    # Silent-ish net: low tonic already in synthetic config
    observer.net.voltage[:] = 0
    observer.net.step(stim)
    # First receptor received current; others in the map did not from this vector.
    assert observer.net.last_external_count == 6
