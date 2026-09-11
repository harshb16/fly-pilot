"""Controller interface.

Only ManualController is implemented in Milestone 1.

Later, real implementations of ExpertLandingController, MaleCNSController,
and TrainedMaleCNSController should land here. Do not add stub classes that
pretend to fly the aircraft.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
        """Ingest the latest authoritative aircraft state.

        Manual control ignores this. A MaleCNS controller would use it
        together with retinal input, which is not present in Milestone 1.
        """

    @abstractmethod
    def act(self) -> AircraftControls:
        raise NotImplementedError
