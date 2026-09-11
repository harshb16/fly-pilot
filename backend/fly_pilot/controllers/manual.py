"""Human-in-the-loop controller.

Browser / keyboard commands are stored here and returned unchanged on act().
JSBSim still applies the physics; this class never integrates motion.
"""

from __future__ import annotations

from fly_pilot.controllers.base import Controller
from fly_pilot.state import AircraftControls, AircraftObservation


class ManualController(Controller):
    def __init__(self, initial: AircraftControls | None = None) -> None:
        self._command = (initial or AircraftControls()).clamped()

    @property
    def name(self) -> str:
        return "manual"

    def reset(self) -> None:
        self._command = AircraftControls()

    def set_pilot_input(self, controls: AircraftControls) -> None:
        self._command = controls.clamped()

    def observe(self, observation: AircraftObservation) -> None:
        return None

    def act(self) -> AircraftControls:
        return self._command
