"""JSBSim Cessna 172P wrapper.

JSBSim is the only source of aircraft motion. This module sets initial
conditions, copies inceptor commands, steps the FDM, and reads telemetry.
"""

from __future__ import annotations

import math
from typing import Any

import jsbsim

from fly_pilot.geodesy import runway_coordinates
from fly_pilot.runway import ApproachConfig, Runway, approach_origin
from fly_pilot.state import AircraftControls, AircraftObservation

FT_TO_M = 0.3048
FPS_TO_FPM = 60.0
FPS_TO_KT = 0.592484
DEG = 180.0 / math.pi

# Pilot elevator: positive = nose up. JSBSim elevator-cmd-norm: positive = nose down.
ELEVATOR_JSBSIM_SIGN = -1.0


class Cessna172:
    """One FGFDMExec instance configured as a Cessna 172P on final."""

    def __init__(
        self,
        runway: Runway | None = None,
        approach: ApproachConfig | None = None,
        dt: float | None = None,
    ) -> None:
        self.runway = runway or Runway()
        self.approach = approach or ApproachConfig()
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model("c172p"):
            raise RuntimeError("JSBSim failed to load aircraft model c172p")
        if dt is not None:
            self.fdm.set_dt(dt)
        self.dt = float(self.fdm.get_delta_t())
        self._last_controls = AircraftControls(throttle=self.approach.throttle)
        self.reset()

    def reset(self) -> AircraftObservation:
        lat, lon, alt_m = approach_origin(self.runway, self.approach)
        alt_ft = alt_m / FT_TO_M
        self.fdm["ic/lat-geod-deg"] = lat
        self.fdm["ic/long-gc-deg"] = lon
        self.fdm["ic/h-sl-ft"] = alt_ft
        self.fdm["ic/terrain-elevation-ft"] = self.runway.alt_m / FT_TO_M
        self.fdm["ic/vc-kts"] = self.approach.airspeed_kts
        self.fdm["ic/psi-true-deg"] = self.runway.heading_deg
        self.fdm["ic/phi-deg"] = 0.0
        self.fdm["ic/gamma-deg"] = self.approach.flight_path_deg
        try:
            self.fdm["ic/alpha-deg"] = self.approach.alpha_deg
        except Exception:
            self.fdm["ic/theta-deg"] = self.approach.flight_path_deg + self.approach.alpha_deg

        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")
        self.fdm.set_sim_time(0.0)

        self._start_engine()
        self._last_controls = AircraftControls(throttle=self.approach.throttle)
        self.apply_controls(self._last_controls)
        return self.observe()

    def _start_engine(self) -> None:
        self.fdm["propulsion/engine/set-running"] = 1.0
        self.fdm["fcs/mixture-cmd-norm"] = self.approach.mixture_norm
        self.fdm["fcs/flap-cmd-norm"] = self.approach.flaps_norm
        self.fdm["fcs/throttle-cmd-norm"] = self.approach.throttle
        self.fdm["gear/gear-cmd-norm"] = 1.0

    def apply_controls(self, controls: AircraftControls) -> None:
        cmd = controls.clamped()
        self._last_controls = cmd
        self.fdm["fcs/aileron-cmd-norm"] = cmd.aileron
        self.fdm["fcs/elevator-cmd-norm"] = ELEVATOR_JSBSIM_SIGN * cmd.elevator
        self.fdm["fcs/rudder-cmd-norm"] = cmd.rudder
        self.fdm["fcs/throttle-cmd-norm"] = cmd.throttle
        self.fdm["fcs/mixture-cmd-norm"] = self.approach.mixture_norm

    def step(self, n: int = 1) -> AircraftObservation:
        if n < 1:
            raise ValueError("n must be >= 1")
        for _ in range(n):
            if not self.fdm.run():
                break
        return self.observe()

    def observe(self) -> AircraftObservation:
        fdm = self.fdm
        lat = float(fdm["position/lat-geod-deg"])
        lon = float(fdm["position/long-gc-deg"])
        alt_msl_m = float(fdm["position/h-sl-ft"]) * FT_TO_M
        alt_agl_m = float(fdm["position/h-agl-ft"]) * FT_TO_M
        east, north, up = self.runway.frame.to_enu(lat, lon, alt_msl_m)
        along, right = runway_coordinates(east, north, self.runway.heading_deg)
        wow = (
            bool(fdm["gear/unit[0]/WOW"]),
            bool(fdm["gear/unit[1]/WOW"]),
            bool(fdm["gear/unit[2]/WOW"]),
        )
        on_ground = bool(fdm["gear/wow"]) or any(wow)
        return AircraftObservation(
            sim_time_s=float(fdm.get_sim_time()),
            lat_deg=lat,
            lon_deg=lon,
            alt_msl_m=alt_msl_m,
            alt_agl_m=alt_agl_m,
            east_m=east,
            north_m=north,
            up_m=up,
            along_m=along,
            right_m=right,
            airspeed_kts=float(fdm["velocities/vc-kts"]),
            groundspeed_kts=float(fdm["velocities/vg-fps"]) * FPS_TO_KT,
            vertical_speed_fpm=float(fdm["velocities/h-dot-fps"]) * FPS_TO_FPM,
            pitch_deg=float(fdm["attitude/pitch-rad"]) * DEG,
            roll_deg=float(fdm["attitude/roll-rad"]) * DEG,
            heading_deg=float(fdm["attitude/psi-deg"]) % 360.0,
            alpha_deg=float(fdm["aero/alpha-deg"]),
            aileron=self._last_controls.aileron,
            elevator=self._last_controls.elevator,
            rudder=self._last_controls.rudder,
            throttle=self._last_controls.throttle,
            on_ground=on_ground,
            wow=wow,
            extra={
                "elevator_pos": float(fdm["fcs/elevator-pos-norm"]),
                "aileron_pos": float(fdm["fcs/left-aileron-pos-norm"]),
                "rudder_pos": float(fdm["fcs/rudder-pos-norm"]),
                "throttle_pos": float(fdm["fcs/throttle-pos-norm"]),
                "beta_deg": float(fdm["aero/beta-deg"]),
            },
        )

    @property
    def raw(self) -> Any:
        return self.fdm
