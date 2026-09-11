"""External current injection into named MaleCNS neurons.

Stimulation is an engineering probe, not a claim about natural sensory drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PulseStimulation:
    """Constant current into a fixed index set on ``[start_step, end_step)``."""

    indices: np.ndarray
    start_step: int
    end_step: int
    amplitude: float
    label: str = "pulse"

    def active(self, step: int) -> bool:
        return self.start_step <= step < self.end_step

    def apply(self, voltage: np.ndarray, step: int) -> int:
        """Add amplitude to the targeted neurons. Returns how many were hit."""
        if not self.active(step) or self.indices.size == 0:
            return 0
        voltage[self.indices] += np.float32(self.amplitude)
        return int(self.indices.size)


@dataclass(frozen=True)
class CurrentStimulation:
    """Per-neuron current for one neural step (retinal drive).

    Amplitudes align with ``indices``. This is the only MaleCNS input used in
    EXPERT + FLY OBSERVING; it never becomes an aircraft command.
    """

    indices: np.ndarray
    amplitudes: np.ndarray
    label: str = "retina"

    def apply(self, voltage: np.ndarray, step: int) -> int:
        if self.indices.size == 0:
            return 0
        voltage[self.indices] += np.asarray(self.amplitudes, dtype=np.float32)
        return int(self.indices.size)
