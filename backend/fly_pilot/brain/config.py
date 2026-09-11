"""Paths, dataset identity, and LIF modeling parameters.

Dataset files are the official MaleCNS v1.0 flat-connectome tables published by
Janelia FlyEM (CC BY 4.0). Hashes below were measured on this project's first
successful download from Google Cloud Storage and are used to reject corrupt
or unexpected files.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

# Official MaleCNS v1.0 aggregated neuron-to-neuron tables.
# Portal: https://male-cns.janelia.org/download/
# Bucket: gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
MALECNS_VERSION = "v1.0"
MALECNS_DATASET = "male-cns:v1.0"
GCS_FLAT_CONNECTOME = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)

SOURCE_FILES: dict[str, dict[str, str | int]] = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "url": f"{GCS_FLAT_CONNECTOME}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        "size_bytes": 14_483_314,
        "md5": "50a7718770c57220f160ba4f431ab89e",
    },
    "transmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "url": f"{GCS_FLAT_CONNECTOME}/body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        "size_bytes": 43_282_834,
        "md5": "3d842b12fe5c49eefade528d7dd24a1f",
    },
    "weights": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "url": f"{GCS_FLAT_CONNECTOME}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        "size_bytes": 1_051_241_946,
        "md5": "f30e9dcca25cfd021bf1e7b3d975599e",
    },
}

# Neurons with a non-empty superclass in the v1.0 annotation table, including
# the 94 `tbc` cells. Status-only filters drop photoreceptors and are not used.
EXPECTED_NEURON_COUNT = 166_700

# Directed edges after requiring both endpoints in the retained neuron set.
# The source Feather table has 151,856,684 rows; most involve unannotated
# fragments. This is not a speed-driven prune.
EXPECTED_EDGE_COUNT = 25_582_938
EXPECTED_WEIGHT_ROWS = 151_856_684

PREPARED_SCHEMA = 1

# Transmitters treated as inhibitory in the point-neuron approximation.
# This follows the Shiu et al. 2024 / Fly64 convention: GABA, glutamate
# (GluCl in adult fly), and histamine (photoreceptor transmitter) as −1.
# Acetylcholine and monoamines (and missing/unclear) are +1. This is a
# modeling approximation, not a synapse-resolved physiological assignment.
INHIBITORY_TRANSMITTERS: frozenset[str] = frozenset({"gaba", "glutamate", "histamine"})

KNOWN_TRANSMITTERS: frozenset[str] = frozenset(
    {
        "acetylcholine",
        "gaba",
        "glutamate",
        "histamine",
        "dopamine",
        "octopamine",
        "serotonin",
        "unclear",
    }
)


def repo_root() -> Path:
    """Return the FlyPilot repository root (parent of ``backend/``)."""
    return Path(__file__).resolve().parents[3]


def default_data_dir() -> Path:
    """Directory for MaleCNS raw downloads and prepared artifacts.

    Override with ``FLYPILOT_MALECNS_DIR``. The default is ``<repo>/data/malecns``,
    which is gitignored except for the README.
    """
    override = os.environ.get("FLYPILOT_MALECNS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data" / "malecns"


def raw_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / "raw"


def prepared_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / "prepared"


@dataclass(frozen=True)
class LIFConfig:
    """Simplified LIF parameters. These are modeling choices, not measurements.

    The 20 ms step (50 neural updates per simulated second) matches the
    simplified MaleCNS demos in Fly64. It is chosen so a later flight loop can
    run several neural steps per JSBSim frame (FDM dt = 1/120 s) without
    claiming a biophysical membrane timebase.
    """

    dt: float = 0.020
    tau_m: float = 0.100
    v_threshold: float = 1.0
    v_reset: float = 0.0
    tonic_current: float = 0.180
    synaptic_gain: float = 1.50
    background_rate_hz: float = 1.2
    background_amplitude: float = 0.22
    normalize_incoming: bool = True
    seed: int = 64

    @property
    def decay(self) -> float:
        return float(math.exp(-self.dt / self.tau_m))

    @property
    def steps_per_simulated_second(self) -> float:
        return 1.0 / self.dt
