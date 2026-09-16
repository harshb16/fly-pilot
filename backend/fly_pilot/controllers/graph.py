"""Task-trained MaleCNS population-graph aircraft controller.

Unlike FLY CONTROL, this policy consumes aircraft telemetry and trains all
neural weights.  Only its directed population topology comes from MaleCNS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fly_pilot.brain.graph_policy import (
    GRAPH_POLICY_KIND,
    GraphPolicyArtifact,
    default_graph_checkpoint_path,
    observation_vector,
)
from fly_pilot.controllers.base import Controller
from fly_pilot.controllers.pid import PID
from fly_pilot.guidance import clamp, glideslope_error_m, heading_error_deg
from fly_pilot.runway import Runway
from fly_pilot.state import AircraftControls, AircraftObservation


class ConnectomeGraphController(Controller):
    """Trainable population graph guidance plus conventional inner loops."""

    name = "hybrid_guidance"
    kind = GRAPH_POLICY_KIND
    biological_learning = False
    uses_expert = False
    uses_aircraft_telemetry = True
    fixed_malecns = False

    def __init__(self, artifact: GraphPolicyArtifact, runway: Runway | None = None) -> None:
        self.artifact = artifact
        self.model = artifact.model
        self.model.eval()
        self.runway = runway or Runway()
        self._observation: AircraftObservation | None = None
        self._previous: AircraftObservation | None = None
        self._controls = AircraftControls(throttle=0.4)
        self._guidance = {
            "roll_command_deg": 0.0,
            "pitch_command_deg": 0.0,
            "target_airspeed_kts": 68.0,
            "throttle_trim": 0.4,
            "phase": "stabilize",
            "graph_roll_command_deg": 0.0,
            "conventional_roll_reference_deg": 0.0,
            "graph_roll_residual_deg": 0.0,
            "graph_roll_clipped": False,
        }
        self._roll_pid = PID(0.04, 0.004, 0.0, -0.7, 0.7, 0.35)
        self._pitch_pid = PID(0.07, 0.012, 0.0, -0.8, 0.8, 0.45)
        self._speed_pid = PID(0.04, 0.01, 0.0, -0.45, 0.45, 0.35)

    @classmethod
    def load(cls, path: Path | None = None, runway: Runway | None = None) -> "ConnectomeGraphController":
        return cls(GraphPolicyArtifact.load(path or default_graph_checkpoint_path()), runway)

    def reset(self) -> None:
        self.model.reset_state()
        self._observation = None
        self._previous = None
        self._controls = AircraftControls(throttle=0.4)
        self._guidance = {
            "roll_command_deg": 0.0,
            "pitch_command_deg": 0.0,
            "target_airspeed_kts": 68.0,
            "throttle_trim": 0.4,
            "phase": "stabilize",
            "graph_roll_command_deg": 0.0,
            "conventional_roll_reference_deg": 0.0,
            "graph_roll_residual_deg": 0.0,
            "graph_roll_clipped": False,
        }
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._speed_pid.reset()

    def observe(self, observation: AircraftObservation) -> None:
        self._previous = self._observation
        self._observation = observation

    def act(self) -> AircraftControls:
        if self._observation is not None:
            values = self.model.step_numpy(observation_vector(self._observation, self.runway))
            graph_roll = float(values[0]) * 25.0
            self._guidance = self._conventional_longitudinal_guidance(self._observation)
            reference_roll = self._conventional_lateral_reference(self._observation)
            deployed_roll = clamp(graph_roll, reference_roll - 4.0, reference_roll + 4.0)
            self._guidance["roll_command_deg"] = deployed_roll
            self._guidance["graph_roll_command_deg"] = graph_roll
            self._guidance["conventional_roll_reference_deg"] = reference_roll
            self._guidance["graph_roll_residual_deg"] = deployed_roll - reference_roll
            self._guidance["graph_roll_clipped"] = abs(deployed_roll - graph_roll) > 1e-6
            self._controls = self._stabilize(self._observation)
        return self._controls

    def _conventional_lateral_reference(self, obs: AircraftObservation) -> float:
        """Safety envelope center; graph guidance may deviate by at most four degrees."""
        cross_track_rate = 0.0
        if self._previous is not None:
            dt = max(obs.sim_time_s - self._previous.sim_time_s, 1e-3)
            cross_track_rate = (obs.right_m - self._previous.right_m) / dt
        phase = self._guidance["phase"]
        heading_limit = 20.0 if phase in ("approach", "stabilize") else 7.0
        heading_offset = clamp(-0.55 * obs.right_m - 0.40 * cross_track_rate, -heading_limit, heading_limit)
        heading_command = self.runway.heading_deg + heading_offset
        heading_error = heading_error_deg(obs.heading_deg, heading_command)
        roll_limit = 18.0 if phase == "stabilize" else (25.0 if phase == "approach" else 7.0)
        return clamp(1.7 * heading_error, -roll_limit, roll_limit)

    def _conventional_longitudinal_guidance(self, obs: AircraftObservation) -> dict[str, float | str]:
        """Explicit non-neural glideslope, airspeed, and flare guidance."""
        if obs.on_ground or obs.alt_agl_m < 3.5:
            return {
                "phase": "touchdown",
                "pitch_command_deg": 3.0,
                "target_airspeed_kts": 0.0,
                "throttle_trim": 0.0,
            }
        if obs.alt_agl_m < 18.0 and obs.along_m > -150.0:
            return {
                "phase": "flare",
                "pitch_command_deg": clamp(2.5 + 0.15 * (18.0 - obs.alt_agl_m), 2.0, 6.0),
                "target_airspeed_kts": 60.0,
                "throttle_trim": 0.08,
            }
        gs_error = glideslope_error_m(obs.alt_agl_m, obs.along_m)
        vertical_speed_command = clamp(50.0 * (-gs_error), -900.0, 400.0)
        return {
            "phase": "stabilize" if obs.sim_time_s < 2.0 else "approach",
            "pitch_command_deg": clamp(-0.5 + 0.010 * (vertical_speed_command - obs.vertical_speed_fpm), -6.0, 7.0),
            "target_airspeed_kts": 70.0 if obs.sim_time_s < 2.0 else 68.0,
            "throttle_trim": 0.38 if gs_error > 8.0 else 0.46,
        }

    def _stabilize(self, obs: AircraftObservation) -> AircraftControls:
        """Conventional rate-damped inner loop; the graph supplies targets."""
        dt = 1.0 / 120.0
        if self._previous is not None:
            dt = max(obs.sim_time_s - self._previous.sim_time_s, 1.0 / 240.0)
        roll_error = self._guidance["roll_command_deg"] - obs.roll_deg
        aileron = self._roll_pid.update(roll_error, dt, deriv=-obs.p_deg_s)
        aileron = clamp(aileron - 0.08 * (obs.p_deg_s * 3.1415926535 / 180.0), -1.0, 1.0)
        pitch_error = self._guidance["pitch_command_deg"] - obs.pitch_deg
        elevator = self._pitch_pid.update(pitch_error, dt, deriv=-obs.q_deg_s)
        elevator = clamp(elevator - 0.12 * (obs.q_deg_s * 3.1415926535 / 180.0), -1.0, 1.0)
        throttle = clamp(
            self._guidance["throttle_trim"]
            + self._speed_pid.update(self._guidance["target_airspeed_kts"] - obs.airspeed_kts, dt),
            0.0,
            0.95,
        )
        if self._guidance["phase"] == "flare":
            throttle = min(throttle, 0.18)
            elevator = clamp(elevator + 0.12 * max(0.0, 65.0 - obs.airspeed_kts) / 15.0, -1.0, 1.0)
        rudder = clamp(
            -0.045 * obs.beta_deg
            - 0.18 * (obs.r_deg_s * 3.1415926535 / 180.0)
            - 0.006 * obs.right_m,
            -0.45,
            0.45,
        )
        return AircraftControls(aileron, elevator, rudder, throttle).clamped()

    def telemetry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": "HYBRID FLY GUIDANCE — safety-bounded connectome-graph bank residual + conventional autoland",
            "male_cns_topology": True,
            "fixed_malecns": False,
            "biological_learning": False,
            "expert_in_loop": False,
            "uses_aircraft_telemetry": True,
            "control_scope": "graph policy contributes a ±4° bank residual inside a conventional lateral envelope; conventional logic controls glideslope, airspeed, flare, and actuator stabilization",
            "population_graph_sha256": self.artifact.graph.sha256,
            "aileron": self._controls.aileron,
            "elevator": self._controls.elevator,
            "rudder": self._controls.rudder,
            "throttle": self._controls.throttle,
            **self._guidance,
        }
