from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.graph_policy import (
    ConnectomeGraphPolicy,
    GraphPolicyArtifact,
    GraphPolicyConfig,
    build_population_graph,
    observation_vector,
    table_observation_matrix,
)
from fly_pilot.controllers.expert import ExpertLandingController
from fly_pilot.controllers.graph import ConnectomeGraphController
from fly_pilot.state import AircraftObservation
from fly_pilot.verify_graph_artifact import verify_graph_artifact


def _observation() -> AircraftObservation:
    return AircraftObservation(
        sim_time_s=12.0,
        lat_deg=37.0,
        lon_deg=-122.0,
        alt_msl_m=180.0,
        alt_agl_m=165.0,
        east_m=0.0,
        north_m=0.0,
        up_m=165.0,
        along_m=-1900.0,
        right_m=35.0,
        airspeed_kts=69.0,
        groundspeed_kts=67.0,
        vertical_speed_fpm=-420.0,
        pitch_deg=-1.0,
        roll_deg=4.0,
        heading_deg=302.0,
        alpha_deg=2.0,
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        throttle=0.4,
        on_ground=False,
        p_deg_s=1.2,
        q_deg_s=-0.5,
        r_deg_s=0.3,
        beta_deg=0.7,
    )


def _connectome() -> Connectome:
    names = np.asarray(
        ["ol_sensory", "cb_sensory", "descending_neuron", "vnc_motor", "vnc_efferent"],
        dtype=object,
    )
    # counts[post, pre]
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 5, 0],
            ],
            dtype=np.float32,
        )
    )
    return Connectome.from_parts(
        np.arange(5, dtype=np.int64),
        counts,
        superclass=names,
        sign=np.asarray([1, -1, 1, 1, 1], dtype=np.int8),
    )


def test_population_graph_and_artifact_roundtrip(tmp_path: Path) -> None:
    graph = build_population_graph(_connectome())
    assert graph.adjacency.shape == (5, 5)
    assert graph.sensory_mask.sum() == 2
    assert graph.readout_mask.sum() == 3
    config = GraphPolicyConfig(node_hidden=8, temporal_hidden=10, message_layers=2)
    model = ConnectomeGraphPolicy(config, graph)
    output, hidden = model(torch.zeros(2, 3, config.observation_dim))
    assert output.shape == (2, 3, 4)
    assert hidden.shape == (1, 2, 10)
    path = tmp_path / "graph.pt"
    GraphPolicyArtifact(
        config,
        graph,
        model,
        {"test": True, "dataset": {}, "training": {}, "source": {}, "metrics": {}},
    ).save(path)
    loaded = GraphPolicyArtifact.load(path)
    assert loaded.graph.sha256 == graph.sha256
    assert loaded.metadata["test"] is True
    report = verify_graph_artifact(path)
    assert report["ok"] is True
    assert report["graph_check"] == "embedded-structural"


def test_vectorized_observation_features_match_live_path() -> None:
    obs = _observation()
    row = {
        "sim_time_s": np.asarray([obs.sim_time_s]),
        "along_m": np.asarray([obs.along_m]),
        "right_m": np.asarray([obs.right_m]),
        "alt_agl_m": np.asarray([obs.alt_agl_m]),
        "airspeed_kts": np.asarray([obs.airspeed_kts]),
        "groundspeed_kts": np.asarray([obs.groundspeed_kts]),
        "vertical_speed_fpm": np.asarray([obs.vertical_speed_fpm]),
        "pitch_deg": np.asarray([obs.pitch_deg]),
        "roll_deg": np.asarray([obs.roll_deg]),
        "heading_deg": np.asarray([obs.heading_deg]),
        "alpha_deg": np.asarray([obs.alpha_deg]),
        "beta_deg": np.asarray([obs.beta_deg]),
        "p_deg_s": np.asarray([obs.p_deg_s]),
        "q_deg_s": np.asarray([obs.q_deg_s]),
        "r_deg_s": np.asarray([obs.r_deg_s]),
    }
    np.testing.assert_allclose(table_observation_matrix(row)[0], observation_vector(obs), rtol=1e-5)


def test_expert_exposes_guidance_targets_separately_from_inceptors() -> None:
    controller = ExpertLandingController()
    controller.observe(_observation())
    controls = controller.act()
    telemetry = controller.telemetry()
    assert -1.0 <= controls.aileron <= 1.0
    assert "roll_command_deg" in telemetry
    assert "pitch_command_deg" in telemetry
    assert "target_airspeed_kts" in telemetry
    assert "throttle_trim" in telemetry
    assert telemetry["kind"] == "conventional_autopilot"


def test_hybrid_graph_guidance_is_safety_bounded() -> None:
    graph = build_population_graph(_connectome())
    config = GraphPolicyConfig(node_hidden=8, temporal_hidden=10, message_layers=1)
    model = ConnectomeGraphPolicy(config, graph)
    artifact = GraphPolicyArtifact(config, graph, model)
    controller = ConnectomeGraphController(artifact)
    controller.reset()
    model.step_numpy = lambda _features: np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]
    controller.observe(_observation())
    controller.act()
    telemetry = controller.telemetry()
    assert telemetry["expert_in_loop"] is False
    assert telemetry["uses_aircraft_telemetry"] is True
    assert abs(telemetry["graph_roll_residual_deg"]) <= 4.0
    assert telemetry["graph_roll_clipped"] is True
