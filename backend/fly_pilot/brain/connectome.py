"""Load the prepared MaleCNS graph as memory-efficient sparse arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

from fly_pilot.brain.config import LIFConfig, default_data_dir, prepared_dir
from fly_pilot.brain.data import DataError, map_body_ids


@dataclass
class Connectome:
    """Measured MaleCNS v1.0 wiring plus neuron annotations.

    ``counts`` is CSR with rows = postsynaptic index, columns = presynaptic
    index, and data = aggregated synapse counts (unsigned). Sign / gain live
    in the LIF layer so the stored graph stays anatomical.
    """

    body_ids: np.ndarray
    counts: sparse.csr_matrix
    type: np.ndarray
    flywire_type: np.ndarray
    instance: np.ndarray
    superclass: np.ndarray
    class_name: np.ndarray
    subclass: np.ndarray
    side: np.ndarray
    consensus_nt: np.ndarray
    sign: np.ndarray
    manifest: dict

    @property
    def n_neurons(self) -> int:
        return int(self.body_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.counts.nnz)

    @property
    def synapse_count_sum(self) -> int:
        return int(self.counts.data.sum())

    def index_of(self, body_id: int) -> int:
        index, valid = map_body_ids(self.body_ids, np.asarray([body_id], dtype=np.int64))
        if not valid[0]:
            raise KeyError(f"body id {body_id} is not in the prepared connectome")
        return int(index[0])

    def indices_of(self, body_ids: np.ndarray) -> np.ndarray:
        index, valid = map_body_ids(self.body_ids, np.asarray(body_ids, dtype=np.int64))
        if not valid.all():
            missing = np.asarray(body_ids)[~valid]
            raise KeyError(f"{len(missing)} body ids are not in the prepared connectome")
        return index

    def counts_nbytes(self) -> int:
        return int(self.counts.data.nbytes + self.counts.indices.nbytes + self.counts.indptr.nbytes)

    def signed_normalized_weights(self, config: LIFConfig | None = None) -> sparse.csr_matrix:
        """Return W[post, pre] for the LIF recurrent term.

        If ``config.normalize_incoming`` is true (default), each postsynaptic
        row is divided by the sum of absolute incoming weights so the LIF
        threshold stays O(1). That normalization is a modeling choice.
        """
        config = config or LIFConfig()
        matrix = self.counts.tocsr().astype(np.float32, copy=True)
        matrix.data *= self.sign[matrix.indices].astype(np.float32)
        if config.normalize_incoming:
            row_counts = np.diff(matrix.indptr)
            rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), row_counts)
            incoming = np.zeros(matrix.shape[0], dtype=np.float32)
            np.add.at(incoming, rows, np.abs(matrix.data))
            matrix.data /= np.maximum(incoming[rows], 1.0)
        return matrix

    @classmethod
    def from_parts(
        cls,
        body_ids: np.ndarray,
        counts: sparse.csr_matrix,
        *,
        type: np.ndarray | None = None,
        flywire_type: np.ndarray | None = None,
        instance: np.ndarray | None = None,
        superclass: np.ndarray | None = None,
        class_name: np.ndarray | None = None,
        subclass: np.ndarray | None = None,
        side: np.ndarray | None = None,
        consensus_nt: np.ndarray | None = None,
        sign: np.ndarray | None = None,
        manifest: dict | None = None,
    ) -> "Connectome":
        n = len(body_ids)
        empty = np.full(n, "", dtype=object)

        def _obj(values: np.ndarray | None) -> np.ndarray:
            if values is None:
                return empty.copy()
            return np.asarray(values, dtype=object)

        if sign is None:
            sign = np.ones(n, dtype=np.int8)
        return cls(
            body_ids=np.asarray(body_ids, dtype=np.int64),
            counts=counts.tocsr().astype(np.float32),
            type=_obj(type),
            flywire_type=_obj(flywire_type),
            instance=_obj(instance),
            superclass=_obj(superclass),
            class_name=_obj(class_name),
            subclass=_obj(subclass),
            side=_obj(side),
            consensus_nt=_obj(consensus_nt),
            sign=np.asarray(sign, dtype=np.int8),
            manifest=manifest or {"synthetic": True},
        )

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Connectome":
        dest = prepared_dir(data_dir or default_data_dir())
        manifest_path = dest / "manifest.json"
        if not manifest_path.exists():
            raise DataError(
                f"Prepared MaleCNS data not found in {dest}. "
                "Run: python -m fly_pilot.brain.prepare"
            )
        manifest = json.loads(manifest_path.read_text())
        table = pq.read_table(dest / "neurons.parquet")
        counts = sparse.load_npz(dest / "weights.npz").astype(np.float32)
        return cls(
            body_ids=table["body_id"].to_numpy(),
            counts=counts.tocsr(),
            type=np.asarray(table["type"].to_pylist(), dtype=object),
            flywire_type=np.asarray(table["flywire_type"].to_pylist(), dtype=object),
            instance=np.asarray(table["instance"].to_pylist(), dtype=object),
            superclass=np.asarray(table["superclass"].to_pylist(), dtype=object),
            class_name=np.asarray(table["class"].to_pylist(), dtype=object),
            subclass=np.asarray(table["subclass"].to_pylist(), dtype=object),
            side=np.asarray(table["side"].to_pylist(), dtype=object),
            consensus_nt=np.asarray(table["consensus_nt"].to_pylist(), dtype=object),
            sign=np.asarray(table["sign"].to_numpy(), dtype=np.int8),
            manifest=manifest,
        )


def require_prepared(data_dir: Path | None = None) -> Path:
    dest = prepared_dir(data_dir or default_data_dir())
    if not (dest / "manifest.json").exists():
        raise DataError(
            f"Prepared MaleCNS data not found in {dest}. "
            "Run: python -m fly_pilot.brain.prepare"
        )
    return dest
