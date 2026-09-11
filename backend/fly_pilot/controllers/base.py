"""Controller interface.

Implemented in this milestone:

- ManualController — human inceptors from the browser
- ExpertLandingController — conventional cascaded PID autoland

ExpertLandingController is classical flight control. It is not MaleCNS.
MaleCNSController and TrainedMaleCNSController are not implemented yet and
must not be stubbed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fly_pilot.state import AircraftControls, AircraftObservation


class Controller(ABC):
    """Map an observation to Cessna inceptors."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear controller-internal state at episode boundaries."""

    def observe(self, observation: AircraftObservation) -> None:
        """Ingest the latest authoritative aircraft state."""

    @abstractmethod
    def act(self) -> AircraftControls:
        raise NotImplementedError

    def telemetry(self) -> dict[str, Any]:
        """Optional debug dict for the HUD. Empty for manual control."""
        return {}
