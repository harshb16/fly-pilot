from pathlib import Path

from fly_pilot.record_observing import synthetic_observer
from fly_pilot.server import SimServer


def test_server_starts_paused() -> None:
    server = SimServer(host="127.0.0.1", port=0)
    assert server.sandbox.paused is True


def test_unavailable_mode_returns_error_and_preserves_connection_state(tmp_path: Path) -> None:
    server = SimServer(host="127.0.0.1", port=0)
    server.sandbox.set_observer(synthetic_observer(seed=80))
    server.sandbox.decoder_path = tmp_path / "missing.pt"
    original = server.sandbox.controller
    response = server._handle_message('{"type":"set_controller","name":"fly_control"}')
    assert response is not None
    assert response["type"] == "error"
    assert response["code"] == "decoder_checkpoint_missing"
    assert server.sandbox.controller is original


def test_missing_malecns_returns_actionable_error(monkeypatch, tmp_path: Path) -> None:
    server = SimServer(host="127.0.0.1", port=0)
    empty = tmp_path / "malecns"
    monkeypatch.setattr("fly_pilot.sandbox.default_data_dir", lambda: empty)
    monkeypatch.setattr(
        "fly_pilot.sandbox.ObservingMaleCNS.load",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("MaleCNS data is not prepared")),
    )
    response = server._handle_message('{"type":"set_controller","name":"expert_observing"}')
    assert response is not None
    assert response["code"] == "malecns_not_prepared"
    assert "brain.prepare" in str(response["action"])
    assert server.sandbox.mode == "manual"


def test_invalid_message_returns_structured_error() -> None:
    server = SimServer(host="127.0.0.1", port=0)
    response = server._handle_message("not-json")
    assert response is not None
    assert response["type"] == "error"
    assert response["code"] == "invalid_message"


def test_unknown_message_type_returns_structured_error() -> None:
    server = SimServer(host="127.0.0.1", port=0)
    response = server._handle_message('{"type":"future_client_command"}')
    assert response is not None
    assert response["code"] == "unsupported_message"
