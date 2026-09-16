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
from fly_pilot.guidance import clamp
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
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._speed_pid.reset()

    def observe(self, observation: AircraftObservation) -> None:
        self._previous = self._observation
        self._observation = observation

    def act(self) -> AircraftControls:
        if self._observation is not None:
            values = self.model.step_numpy(observation_vector(self._observation, self.runway))
            self._guidance = {
                "roll_command_deg": float(values[0]) * 25.0,
                "pitch_command_deg": float(values[1]) * 8.0,
                "target_airspeed_kts": 65.0 + float(values[2]) * 15.0,
                "throttle_trim": float(values[3]) * 0.60,
            }
            self._controls = self._stabilize(self._observation)
        return self._controls

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
            "label": "HYBRID FLY GUIDANCE — task-trained MaleCNS topology + conventional stabilization",
            "male_cns_topology": True,
            "fixed_malecns": False,
            "biological_learning": False,
            "expert_in_loop": False,
            "uses_aircraft_telemetry": True,
            "control_scope": "graph policy sets roll/pitch/airspeed/throttle targets; conventional PID tracks them",
            "population_graph_sha256": self.artifact.graph.sha256,
            "aileron": self._controls.aileron,
            "elevator": self._controls.elevator,
            "rudder": self._controls.rudder,
            "throttle": self._controls.throttle,
            **self._guidance,
        }
