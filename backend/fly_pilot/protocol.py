"""JSON protocol between the browser and the Python sandbox."""

from __future__ import annotations

import json
from typing import Any

from fly_pilot.runway import Runway
from fly_pilot.sandbox import LandingSandbox, SandboxSnapshot
from fly_pilot.state import AircraftControls


def parse_client_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "type" not in data:
        raise ValueError("client message must be an object with a type field")
    return data


def controls_from_message(data: dict[str, Any]) -> AircraftControls:
    return AircraftControls(
        aileron=float(data.get("aileron", 0.0)),
        elevator=float(data.get("elevator", 0.0)),
        rudder=float(data.get("rudder", 0.0)),
        throttle=float(data.get("throttle", 0.4)),
    ).clamped()


def error_payload(code: str, message: str, action: str | None = None) -> dict[str, Any]:
    return {
        "type": "error",
        "code": code,
        "message": message,
        "action": action,
    }


def hello_payload(sandbox: LandingSandbox) -> dict[str, Any]:
    runway: Runway = sandbox.runway
    approach = sandbox.approach
    mode = getattr(sandbox, "mode", sandbox.controller.name)
    observing = bool(getattr(sandbox, "observing", False))
    if mode == "fly_control":
        note = (
            "FLY CONTROL: fixed MaleCNS LIF plus a trained temporal decoder writes "
            "JSBSim inceptors. This is not biological synaptic learning. "
            "ExpertLandingController is not in the loop. Decoder input is DN activity only."
        )
    elif mode == "expert_observing":
        note = (
            "EXPERT + FLY OBSERVING: ExpertLandingController (conventional autopilot) "
            "flies the Cessna. MaleCNS watches a rendered fly-view. The fly is NOT "
            "controlling the aircraft."
        )
    elif sandbox.controller.name == "expert":
        note = (
            "ExpertLandingController is a conventional classical autopilot used as a "
            "solvability baseline and expert-data generator. It is not MaleCNS and is "
            "not biological computation."
        )
    else:
        note = (
            "Milestone 5 landing sandbox. MANUAL and EXPERT are unchanged. "
            "FLY CONTROL uses frozen MaleCNS plus an external trained decoder."
        )
    fly_controls = mode == "fly_control"
    payload: dict[str, Any] = {
        "type": "hello",
        "milestone": 5,
        "controller": mode,
        "control_authority": sandbox.controller.name,
        "physics": "jsbsim",
        "aircraft": "c172p",
        "runway": {
            "name": runway.name,
            "lat_deg": runway.lat_deg,
            "lon_deg": runway.lon_deg,
            "alt_m": runway.alt_m,
            "heading_deg": runway.heading_deg,
            "length_m": runway.length_m,
            "width_m": runway.width_m,
        },
        "approach": {
            "distance_m": approach.distance_m,
            "agl_m": approach.agl_m,
            "airspeed_kts": approach.airspeed_kts,
            "flight_path_deg": approach.flight_path_deg,
        },
        "integrity": {
            "authoritative_physics": "JSBSim Cessna 172P",
            "visuals": "Three.js human view; Python cubemap is canonical MaleCNS input",
            "male_cns": fly_controls or observing,
            "male_cns_observing": observing,
            "fly_controls_aircraft": fly_controls,
            "expert_is_biological": False,
            "biological_learning": False,
            "decoder_input": "dn_windowed_rates" if fly_controls else None,
            "note": note,
        },
        "schedule": {
            "physics_hz": 120.0,
            "vision_hz": 50.0,
            "neural_hz": 50.0,
            "state_broadcast_hz": 30.0,
        },
        "capabilities": (
            sandbox.mode_capabilities()
            if callable(getattr(sandbox, "mode_capabilities", None))
            else {}
        ),
    }
    observer = getattr(sandbox, "observer", None)
    if observer is not None and (observing or fly_controls):
        payload["observing"] = observer.metadata()
        payload["observing"]["controls_aircraft"] = fly_controls
    return payload


def state_payload(snapshot: SandboxSnapshot) -> dict[str, Any]:
    obs = snapshot.observation
    ep = snapshot.episode
    extra = obs.extra or {}
    mode = snapshot.ui_mode
    payload: dict[str, Any] = {
        "type": "state",
        "sim_time": obs.sim_time_s,
        "paused": snapshot.paused,
        "controller": mode,
        "control_authority": snapshot.controller_name,
        "observing": snapshot.observing,
        "spawn_seed": snapshot.spawn_seed,
        "position": {
            "lat_deg": obs.lat_deg,
            "lon_deg": obs.lon_deg,
            "alt_msl_m": obs.alt_msl_m,
            "alt_agl_m": obs.alt_agl_m,
            "east_m": obs.east_m,
            "north_m": obs.north_m,
            "up_m": obs.up_m,
            "along_m": obs.along_m,
            "right_m": obs.right_m,
        },
        "velocity": {
            "airspeed_kts": obs.airspeed_kts,
            "groundspeed_kts": obs.groundspeed_kts,
            "vertical_speed_fpm": obs.vertical_speed_fpm,
        },
        "attitude": {
            "pitch_deg": obs.pitch_deg,
            "roll_deg": obs.roll_deg,
            "heading_deg": obs.heading_deg,
            "alpha_deg": obs.alpha_deg,
            "p_deg_s": obs.p_deg_s,
            "q_deg_s": obs.q_deg_s,
            "r_deg_s": obs.r_deg_s,
            "beta_deg": obs.beta_deg,
        },
        "controls": {
            "aileron": obs.aileron,
            "elevator": obs.elevator,
            "rudder": obs.rudder,
            "throttle": obs.throttle,
        },
        "surfaces": {
            "elevator_pos": extra.get("elevator_pos", obs.elevator),
            "aileron_pos": extra.get("aileron_pos", obs.aileron),
            "rudder_pos": extra.get("rudder_pos", obs.rudder),
            "throttle_pos": extra.get("throttle_pos", obs.throttle),
        },
        "gear": {
            "on_ground": obs.on_ground,
            "wow": list(obs.wow),
        },
        "episode": {
            "status": ep.status.value,
            "reason": ep.reason,
            "touchdown_fpm": ep.touchdown_fpm,
        },
    }
    if snapshot.controller_name == "expert" and snapshot.debug:
        payload["expert"] = snapshot.debug
    if snapshot.ui_mode == "fly_control":
        payload["fly_control"] = {
            k: snapshot.debug[k]
            for k in (
                "kind",
                "label",
                "male_cns",
                "biological_learning",
                "expert_in_loop",
                "aileron",
                "elevator",
                "rudder",
                "throttle",
                "gru_hidden_norm",
                "gru_hidden_mean",
                "decoded_neural_step",
                "slew_alpha",
            )
            if k in snapshot.debug
        }
    if snapshot.observing or snapshot.ui_mode == "fly_control":
        payload["fly_observing"] = snapshot.fly_observing
    return payload
