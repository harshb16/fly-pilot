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


def hello_payload(sandbox: LandingSandbox) -> dict[str, Any]:
    runway: Runway = sandbox.runway
    approach = sandbox.approach
    return {
        "type": "hello",
        "milestone": 1,
        "controller": sandbox.controller.name,
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
            "visuals": "Three.js rendering of JSBSim state",
            "male_cns": False,
            "note": "Milestone 1 is a human-flown landing sandbox. The MaleCNS connectome is not in the loop.",
        },
    }


def state_payload(snapshot: SandboxSnapshot) -> dict[str, Any]:
    obs = snapshot.observation
    ep = snapshot.episode
    return {
        "type": "state",
        "sim_time": obs.sim_time_s,
        "paused": snapshot.paused,
        "controller": snapshot.controller_name,
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
        },
        "controls": {
            "aileron": obs.aileron,
            "elevator": obs.elevator,
            "rudder": obs.rudder,
            "throttle": obs.throttle,
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
