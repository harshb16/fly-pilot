import json

from fly_pilot.protocol import controls_from_message, hello_payload, parse_client_message, state_payload
from fly_pilot.runway import Runway
from fly_pilot.sandbox import SandboxSnapshot
from fly_pilot.state import AircraftControls, AircraftObservation, EpisodeInfo, EpisodeStatus


def test_parse_and_clamp_controls() -> None:
    msg = parse_client_message('{"type":"controls","aileron":1.5,"elevator":-0.2,"rudder":0,"throttle":0.7}')
    assert msg["type"] == "controls"
    cmd = controls_from_message(msg)
    assert cmd.aileron == 1.0
    assert cmd.elevator == -0.2
    assert cmd.throttle == 0.7


def test_state_payload_contains_required_fields() -> None:
    obs = AircraftObservation(
        sim_time_s=1.25,
        lat_deg=37.0,
        lon_deg=-122.01,
        alt_msl_m=250.0,
        alt_agl_m=250.0,
        east_m=-3000.0,
        north_m=0.0,
        up_m=250.0,
        along_m=-3000.0,
        right_m=0.0,
        airspeed_kts=70.0,
        groundspeed_kts=69.0,
        vertical_speed_fpm=-420.0,
        pitch_deg=-1.5,
        roll_deg=0.4,
        heading_deg=90.0,
        alpha_deg=4.0,
        aileron=0.1,
        elevator=0.2,
        rudder=-0.05,
        throttle=0.42,
        on_ground=False,
    )
    payload = state_payload(
        SandboxSnapshot(obs, EpisodeInfo(), "manual", False)
    )
    assert payload["type"] == "state"
    for key in (
        "position",
        "velocity",
        "attitude",
        "controls",
        "gear",
        "episode",
    ):
        assert key in payload
    dumped = json.dumps(payload)
    assert "250.0" in dumped
    assert payload["controls"]["elevator"] == 0.2
    assert payload["episode"]["status"] == EpisodeStatus.IN_PROGRESS.value


def test_hello_payload_states_no_malecns() -> None:
    # Avoid constructing JSBSim here: build a tiny duck-typed sandbox-like object.
    class _Fake:
        runway = Runway()
        approach = type("A", (), {"distance_m": 3200, "agl_m": 250, "airspeed_kts": 70, "flight_path_deg": -4.5})()
        controller = type("C", (), {"name": "manual"})()

    payload = hello_payload(_Fake())  # type: ignore[arg-type]
    assert payload["male_cns"] is False if "male_cns" in payload else payload["integrity"]["male_cns"] is False
    assert payload["physics"] == "jsbsim"
