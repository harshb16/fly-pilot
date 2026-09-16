"""Standalone MaleCNS-derived spiking-network simulation.

This package loads the measured MaleCNS v1.0 connectome and runs a simplified
leaky-integrate-and-fire model on that anatomy. Milestone 5 adds an external
trained temporal decoder on descending-neuron activity. MaleCNS synapses are
not trained. The decoder is not biological learning.

Scientific distinction (keep this wording):

- Connectivity comes from measured MaleCNS anatomy.
- Neural firing dynamics are a modeling choice, not a validated biophysical
  model of a living fruit fly.

Do not describe this module as an exact fly brain, a living-fly simulation,
biologically validated neural dynamics, or proof of fly cognition.
"""

from fly_pilot.brain.config import LIFConfig, default_data_dir
from fly_pilot.brain.connectome import Connectome
from fly_pilot.brain.model import MaleCNSLIF
from fly_pilot.brain.populations import NeuronIndex

__all__ = [
    "Connectome",
    "LIFConfig",
    "MaleCNSLIF",
    "NeuronIndex",
    "default_data_dir",
]
