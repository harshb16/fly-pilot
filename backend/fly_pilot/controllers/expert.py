"""Conventional (non-biological) expert landing controller.

ExpertLandingController is ordinary cascaded classical flight control for the
JSBSim Cessna 172P. It exists only as:

1. a solvability baseline for the landing environment,
2. an expert-demonstration generator for a later decoder,
3. a comparison benchmark for TrainedMaleCNSController.

It is NOT MaleCNS, NOT a fly brain, and MUST NOT be presented as biological
computation. It must never be blended into FLY CONTROL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fly_pilot.controllers.base import Controller
from fly_pilot.controllers.pid import PID
from fly_pilot.guidance import (
    DEFAULT_AIM_ALONG_M,
    DEFAULT_GLIDESLOPE_DEG,
    clamp,
    glideslope_error_deg,
    glideslope_error_m,
    heading_error_deg,
)
from fly_pilot.runway import Runway
from fly_pilot.state import AircraftControls, AircraftObservation


class LandingPhase(str, Enum):
    """Human-readable phases. Ordering is chronological on a typical approach."""

    STABILIZE = "stabilize"
    APPROACH = "approach"
    FLARE = "flare"
    TOUCHDOWN = "touchdown"
    ROLLOUT = "rollout"


@dataclass(frozen=True, slots=True)
class ExpertLandingGains:
    """Tuned against live JSBSim c172p (see docs/expert-controller.md)."""

    aim_along_m: float = DEFAULT_AIM_ALONG_M
    glideslope_deg: float = DEFAULT_GLIDESLOPE_DEG
    approach_ias_kts: float = 68.0
    stabilize_ias_kts: float = 70.0
    flare_ias_kts: float = 60.0
    flare_agl_m: float = 18.0
    flare_along_m: float = -150.0
    touchdown_agl_m: float = 3.5
    stabilize_s: float = 2.0
    xtk_gain: float = 0.55
    xtk_dot_gain: float = 0.40
    heading_to_roll: float = 1.7
    gs_vs_gain: float = 50.0
    vs_to_pitch: float = 0.010
    pitch_bias_deg: float = -0.5
    approach_throttle_trim: float = 0.38
    high_gs_throttle_trim: float = 0.46
    beta_rudder: float = -0.045
    yaw_damper: float = -0.18
    roll_rate_aileron: float = -0.08
    pitch_rate_elevator: float = -0.12


class ExpertLandingController(Controller):
    """Cascaded PID autoland. Classical aerospace, not a fly connectome."""

    def __init__(
        self,
        runway: Runway | None = None,
        gains: ExpertLandingGains | None = None,
    ) -> None:
        self.runway = runway or Runway()
        self.gains = gains or ExpertLandingGains()
        self.phase = LandingPhase.STABILIZE
        self._obs: AircraftObservation | None = None
        self._prev: AircraftObservation | None = None
        self._time_s = 0.0
        self._roll_pid = PID(0.04, 0.004, 0.0, -0.7, 0.7, 0.35)
        self._pitch_pid = PID(0.07, 0.012, 0.0, -0.8, 0.8, 0.45)
        self._speed_pid = PID(0.04, 0.01, 0.0, -0.45, 0.45, 0.35)
        self._telemetry: dict[str, float | str | bool] = {}
        self._guidance: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "expert"

    @property
    def kind(self) -> str:
        return "conventional_autopilot"

    def reset(self) -> None:
        self.phase = LandingPhase.STABILIZE
        self._obs = None
        self._prev = None
        self._time_s = 0.0
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._speed_pid.reset()
        self._telemetry = {}
        self._guidance = {}

    def observe(self, observation: AircraftObservation) -> None:
        self._prev = self._obs
        self._obs = observation
        self._time_s = observation.sim_time_s

    def act(self) -> AircraftControls:
        obs = self._obs
        if obs is None:
            return AircraftControls(throttle=0.4).clamped()
        dt = self._dt(obs)
        self.phase = self._select_phase(obs)
        controls = self._compute(obs, dt)
        self._fill_telemetry(obs, controls)
        return controls.clamped()

    def telemetry(self) -> dict[str, float | str | bool]:
        """HUD / logging snapshot. Always labelled as a conventional autopilot."""
        return dict(self._telemetry)

    def _dt(self, obs: AircraftObservation) -> float:
        if self._prev is None:
            return 1.0 / 120.0
        return max(obs.sim_time_s - self._prev.sim_time_s, 1.0 / 240.0)

    def _select_phase(self, obs: AircraftObservation) -> LandingPhase:
        if obs.on_ground:
            return LandingPhase.ROLLOUT
        if obs.alt_agl_m < self.gains.touchdown_agl_m:
            return LandingPhase.TOUCHDOWN
        if obs.alt_agl_m < self.gains.flare_agl_m and obs.along_m > self.gains.flare_along_m:
            return LandingPhase.FLARE
        if obs.sim_time_s < self.gains.stabilize_s:
            return LandingPhase.STABILIZE
        return LandingPhase.APPROACH

    def _compute(self, obs: AircraftObservation, dt: float) -> AircraftControls:
        g = self.gains
        rwy_hdg = self.runway.heading_deg
        gs_err_m = glideslope_error_m(
            obs.alt_agl_m, obs.along_m, aim_along_m=g.aim_along_m, glideslope_deg=g.glideslope_deg
        )
        xtk_dot = 0.0
        if self._prev is not None:
            dtp = max(obs.sim_time_s - self._prev.sim_time_s, 1e-3)
            xtk_dot = (obs.right_m - self._prev.right_m) / dtp

        aileron, rudder = self._lateral(obs, dt, rwy_hdg, xtk_dot)
        elevator, throttle = self._longitudinal(obs, dt, gs_err_m)
        return AircraftControls(aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle)

    def _lateral(
        self, obs: AircraftObservation, dt: float, rwy_hdg: float, xtk_dot: float
    ) -> tuple[float, float]:
        g = self.gains
        phase = self.phase
        hdg_off_lim = 20.0 if phase in (LandingPhase.APPROACH, LandingPhase.STABILIZE) else 7.0
        hdg_off = clamp(-g.xtk_gain * obs.right_m - g.xtk_dot_gain * xtk_dot, -hdg_off_lim, hdg_off_lim)
        hdg_cmd = rwy_hdg + hdg_off
        hdg_err = heading_error_deg(obs.heading_deg, hdg_cmd)
        if phase is LandingPhase.APPROACH:
            roll_lim = 25.0
        elif phase is LandingPhase.STABILIZE:
            roll_lim = 18.0
        else:
            roll_lim = 7.0
        roll_cmd = 0.0 if phase is LandingPhase.ROLLOUT else clamp(g.heading_to_roll * hdg_err, -roll_lim, roll_lim)
        self._guidance["roll_command_deg"] = roll_cmd
        roll_err = roll_cmd - obs.roll_deg
        aileron = self._roll_pid.update(roll_err, dt, deriv=-obs.p_deg_s)
        aileron = clamp(aileron + g.roll_rate_aileron * (obs.p_deg_s * 3.1415926535 / 180.0), -1.0, 1.0)

        rudder = clamp(
            g.beta_rudder * obs.beta_deg + g.yaw_damper * (obs.r_deg_s * 3.1415926535 / 180.0) - 0.006 * obs.right_m,
            -0.45,
            0.45,
        )
        if phase is LandingPhase.ROLLOUT:
            track_err = heading_error_deg(obs.heading_deg, rwy_hdg)
            rudder = clamp(0.05 * track_err - 0.03 * obs.right_m, -0.7, 0.7)
            aileron = clamp(-0.04 * obs.roll_deg, -0.5, 0.5)
        return aileron, rudder

    def _longitudinal(
        self, obs: AircraftObservation, dt: float, gs_err_m: float
    ) -> tuple[float, float]:
        g = self.gains
        phase = self.phase
        if phase is LandingPhase.FLARE:
            agl = obs.alt_agl_m
            pitch_cmd = clamp(2.5 + 0.15 * (g.flare_agl_m - agl), 2.0, 6.0)
            ias_cmd = g.flare_ias_kts
            thr_trim = 0.08
        elif phase in (LandingPhase.TOUCHDOWN, LandingPhase.ROLLOUT):
            self._guidance.update(
                pitch_command_deg=3.0,
                target_airspeed_kts=0.0,
                throttle_trim=0.0,
            )
            elevator = clamp(0.18 - 0.02 * obs.pitch_deg, 0.05, 0.45)
            return elevator, 0.0
        else:
            vs_cmd = clamp(g.gs_vs_gain * (-gs_err_m), -900.0, 400.0)
            pitch_cmd = clamp(
                g.pitch_bias_deg + g.vs_to_pitch * (vs_cmd - obs.vertical_speed_fpm),
                -6.0,
                7.0,
            )
            ias_cmd = g.stabilize_ias_kts if phase is LandingPhase.STABILIZE else g.approach_ias_kts
            thr_trim = g.approach_throttle_trim if gs_err_m > 8.0 else g.high_gs_throttle_trim

        self._guidance.update(
            pitch_command_deg=pitch_cmd,
            target_airspeed_kts=ias_cmd,
            throttle_trim=thr_trim,
        )

        pitch_err = pitch_cmd - obs.pitch_deg
        elevator = self._pitch_pid.update(pitch_err, dt, deriv=-obs.q_deg_s)
        elevator = clamp(
            elevator + g.pitch_rate_elevator * (obs.q_deg_s * 3.1415926535 / 180.0),
            -1.0,
            1.0,
        )
        throttle = clamp(thr_trim + self._speed_pid.update(ias_cmd - obs.airspeed_kts, dt), 0.0, 0.95)
        if phase is LandingPhase.FLARE:
            throttle = min(throttle, 0.18)
            # Extra back-stick as speed bleeds off in the flare.
            elevator = clamp(elevator + 0.12 * max(0.0, 65.0 - obs.airspeed_kts) / 15.0, -1.0, 1.0)
        return elevator, throttle

    def _fill_telemetry(self, obs: AircraftObservation, controls: AircraftControls) -> None:
        g = self.gains
        gs_m = glideslope_error_m(
            obs.alt_agl_m, obs.along_m, aim_along_m=g.aim_along_m, glideslope_deg=g.glideslope_deg
        )
        gs_deg = glideslope_error_deg(
            obs.alt_agl_m, obs.along_m, aim_along_m=g.aim_along_m, glideslope_deg=g.glideslope_deg
        )
        target_ias = {
            LandingPhase.STABILIZE: g.stabilize_ias_kts,
            LandingPhase.APPROACH: g.approach_ias_kts,
            LandingPhase.FLARE: g.flare_ias_kts,
            LandingPhase.TOUCHDOWN: 0.0,
            LandingPhase.ROLLOUT: 0.0,
        }[self.phase]
        self._telemetry = {
            "kind": "conventional_autopilot",
            "label": "ExpertLandingController — classical autopilot, not biological",
            "male_cns": False,
            "phase": self.phase.value,
            "target_airspeed_kts": target_ias,
            "glideslope_error_deg": gs_deg,
            "glideslope_error_m": gs_m,
            "centerline_error_m": obs.right_m,
            "heading_error_deg": heading_error_deg(obs.heading_deg, self.runway.heading_deg),
            "aileron": controls.aileron,
            "elevator": controls.elevator,
            "rudder": controls.rudder,
            "throttle": controls.throttle,
            **self._guidance,
        }
