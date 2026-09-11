"""Episode outcomes for the landing sandbox.

Termination is evaluated from JSBSim state against runway geometry.
None of these rules move the aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass

from fly_pilot.runway import ApproachConfig, Runway
from fly_pilot.state import AircraftObservation, EpisodeInfo, EpisodeStatus


@dataclass
class EpisodeRules:
    max_time_s: float = 180.0
    max_lateral_m: float = 900.0
    max_agl_m: float = 700.0
    past_runway_m: float = 250.0
    max_distance_m: float = 9000.0
    landing_max_sink_fpm: float = 700.0
    landing_max_roll_deg: float = 12.0
    landing_max_drift_m: float = 8.0
    crash_sink_fpm: float = 900.0
    crash_roll_deg: float = 40.0
    crash_pitch_deg: float = 25.0
    off_runway_margin_m: float = 10.0


class EpisodeMonitor:
    def __init__(
        self,
        runway: Runway | None = None,
        approach: ApproachConfig | None = None,
        rules: EpisodeRules | None = None,
    ) -> None:
        self.runway = runway or Runway()
        self.approach = approach or ApproachConfig()
        self.rules = rules or EpisodeRules()
        self.info = EpisodeInfo()
        self._was_airborne = True

    def reset(self) -> EpisodeInfo:
        self.info = EpisodeInfo()
        self._was_airborne = True
        return self.info

    def update(self, obs: AircraftObservation) -> EpisodeInfo:
        if self.info.status is not EpisodeStatus.IN_PROGRESS:
            return self.info

        airborne = not obs.on_ground
        if airborne:
            self._was_airborne = True

        crash = self._crash_reason(obs)
        if crash:
            self.info = EpisodeInfo(EpisodeStatus.CRASHED, crash, obs.vertical_speed_fpm)
            return self.info

        if obs.on_ground and self._was_airborne and self._is_good_touchdown(obs):
            self.info = EpisodeInfo(
                EpisodeStatus.LANDED,
                "touchdown on runway within sink/attitude limits",
                obs.vertical_speed_fpm,
            )
            return self.info

        failed = self._failed_approach_reason(obs)
        if failed:
            self.info = EpisodeInfo(EpisodeStatus.FAILED_APPROACH, failed, obs.vertical_speed_fpm)
            return self.info

        oob = self._out_of_bounds_reason(obs)
        if oob:
            self.info = EpisodeInfo(EpisodeStatus.OUT_OF_BOUNDS, oob, obs.vertical_speed_fpm)
            return self.info

        return self.info

    def _over_runway(self, obs: AircraftObservation, extra_width: float = 0.0) -> bool:
        half = self.runway.width_m / 2.0 + extra_width
        return 0.0 <= obs.along_m <= self.runway.length_m and abs(obs.right_m) <= half

    def _is_good_touchdown(self, obs: AircraftObservation) -> bool:
        sink = -obs.vertical_speed_fpm
        return (
            self._over_runway(obs, extra_width=self.rules.landing_max_drift_m)
            and sink <= self.rules.landing_max_sink_fpm
            and abs(obs.roll_deg) <= self.rules.landing_max_roll_deg
            and abs(obs.pitch_deg) <= 12.0
        )

    def _crash_reason(self, obs: AircraftObservation) -> str | None:
        if obs.on_ground and -obs.vertical_speed_fpm > self.rules.crash_sink_fpm:
            return f"hard landing ({obs.vertical_speed_fpm:.0f} fpm)"
        if obs.on_ground and not self._over_runway(obs, extra_width=self.rules.off_runway_margin_m):
            return "ground contact off the runway"
        if obs.alt_agl_m < 12.0 and abs(obs.roll_deg) > self.rules.crash_roll_deg:
            return "excessive bank near the ground"
        if obs.alt_agl_m < 10.0 and obs.pitch_deg < -self.rules.crash_pitch_deg:
            return "nose-low attitude near the ground"
        if obs.alt_agl_m < 2.0 and not obs.on_ground and -obs.vertical_speed_fpm > self.rules.crash_sink_fpm:
            return "impacted terrain"
        return None

    def _failed_approach_reason(self, obs: AircraftObservation) -> str | None:
        if obs.along_m > self.runway.length_m + self.rules.past_runway_m and obs.alt_agl_m > 40.0:
            return "flew past the runway without landing"
        if obs.on_ground and self._over_runway(obs) and -obs.vertical_speed_fpm > self.rules.landing_max_sink_fpm:
            return "touchdown sink rate exceeded landing limits"
        return None

    def _out_of_bounds_reason(self, obs: AircraftObservation) -> str | None:
        if obs.sim_time_s > self.rules.max_time_s:
            return "episode time limit"
        if abs(obs.right_m) > self.rules.max_lateral_m:
            return "excessive lateral deviation"
        if obs.alt_agl_m > self.rules.max_agl_m:
            return "climbed above approach ceiling"
        range_m = math_hypot(obs.east_m, obs.north_m)
        if range_m > self.rules.max_distance_m:
            return "too far from the runway"
        if obs.along_m < -self.approach.distance_m - 600.0:
            return "backed away from the approach"
        return None


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5
