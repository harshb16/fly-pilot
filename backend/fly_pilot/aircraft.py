"""JSBSim Cessna 172P wrapper.

JSBSim is the only source of aircraft motion. This module sets initial
conditions, copies inceptor commands, steps the FDM, and reads telemetry.
"""

from __future__ import annotations

import math
from typing import Any

import jsbsim

from fly_pilot.geodesy import runway_coordinates
from fly_pilot.initial_conditions import SpawnState, spawn_geodetic
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

    def reset(self, spawn: SpawnState | None = None) -> AircraftObservation:
        """Place the C172 on short final and start the engine.

        JSBSim 1.3.1 c172p ignores `propulsion/engine/set-running = 1` until the
        Lycoming has actually been cranked. We crank, then `run_ic()` again so
        the episode still begins at the approach spawn (engine stays running).
        """
        self._apply_ics(spawn)
        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed")
        self._crank_engine()
        self._apply_ics(spawn)
        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim run_ic() failed after engine start")
        self.fdm.set_sim_time(0.0)
        self._configure_systems()
        self._last_controls = AircraftControls(throttle=self.approach.throttle)
        self.apply_controls(self._last_controls)
        return self.observe()

    def _apply_ics(self, spawn: SpawnState | None) -> None:
        if spawn is None:
            lat, lon, alt_m = approach_origin(self.runway, self.approach)
            heading = self.runway.heading_deg
            roll = 0.0
            vc = self.approach.airspeed_kts
            gamma = self.approach.flight_path_deg
            alpha = self.approach.alpha_deg
        else:
            lat, lon, alt_m = spawn_geodetic(self.runway, spawn)
            heading = spawn.heading_deg
            roll = spawn.roll_deg
            vc = spawn.airspeed_kts
            gamma = spawn.gamma_deg
            alpha = spawn.alpha_deg
        self.fdm["ic/lat-geod-deg"] = lat
        self.fdm["ic/long-gc-deg"] = lon
        self.fdm["ic/h-sl-ft"] = alt_m / FT_TO_M
        self.fdm["ic/terrain-elevation-ft"] = self.runway.alt_m / FT_TO_M
        self.fdm["ic/vc-kts"] = vc
        self.fdm["ic/psi-true-deg"] = heading
        self.fdm["ic/phi-deg"] = roll
        self.fdm["ic/gamma-deg"] = gamma
        try:
            self.fdm["ic/alpha-deg"] = alpha
        except Exception:
            self.fdm["ic/theta-deg"] = gamma + alpha

    def _crank_engine(self) -> None:
        """Starter + magnetos until the piston engine catches.

        Writing `set-running = 1` without cranking leaves RPM=0 and thrust=0,
        which is why an uncommanded C172 appears to sink with a dead engine.
        """
        fdm = self.fdm
        fdm["fcs/mixture-cmd-norm"] = 1.0
        fdm["fcs/throttle-cmd-norm"] = 0.2
        fdm["gear/gear-cmd-norm"] = 1.0
        caught = False
        n = int(4.0 / max(self.dt, 1e-3))
        for _ in range(n):
            fdm["propulsion/starter_cmd"] = 1.0
            fdm["propulsion/magneto_cmd"] = 3.0
            fdm.run()
            rpm = float(fdm["propulsion/engine/engine-rpm"])
            running = float(fdm["propulsion/engine/set-running"])
            if running >= 1.0 and rpm > 1000.0:
                caught = True
                break
        fdm["propulsion/starter_cmd"] = 0.0
        fdm["propulsion/magneto_cmd"] = 3.0
        if not caught:
            raise RuntimeError("JSBSim c172p engine failed to start")

    def _configure_systems(self) -> None:
        self.fdm["propulsion/magneto_cmd"] = 3.0
        self.fdm["fcs/mixture-cmd-norm"] = self.approach.mixture_norm
        self.fdm["fcs/flap-cmd-norm"] = self.approach.flaps_norm
        self.fdm["fcs/throttle-cmd-norm"] = self.approach.throttle
        self.fdm["gear/gear-cmd-norm"] = 1.0

    def _start_engine(self) -> None:
        # Kept for callers/tests that used the old name; crank happens in reset().
        self._configure_systems()

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
            p_deg_s=float(fdm["velocities/p-rad_sec"]) * DEG,
            q_deg_s=float(fdm["velocities/q-rad_sec"]) * DEG,
            r_deg_s=float(fdm["velocities/r-rad_sec"]) * DEG,
            beta_deg=float(fdm["aero/beta-deg"]),
            extra={
                "elevator_pos": float(fdm["fcs/elevator-pos-norm"]),
                "aileron_pos": float(fdm["fcs/left-aileron-pos-norm"]),
                "rudder_pos": float(fdm["fcs/rudder-pos-norm"]),
                "throttle_pos": float(fdm["fcs/throttle-pos-norm"]),
                "beta_deg": float(fdm["aero/beta-deg"]),
                "engine_rpm": float(fdm["propulsion/engine/engine-rpm"]),
                "thrust_lbs": float(fdm["propulsion/engine/thrust-lbs"]),
            },
        )

    @property
    def raw(self) -> Any:
        return self.fdm
