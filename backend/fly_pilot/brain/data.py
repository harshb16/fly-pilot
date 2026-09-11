"""Download, validate, and compact MaleCNS v1.0 connectivity.

The source tables are already neuron-to-neuron (not the 100M+ raw synapse
list). Preparation:

1. Download the three official Feather files if missing or hash-mismatched.
2. Keep every neuron with a non-empty superclass (166,700 cells).
3. Map directed edges whose both endpoints are in that set.
4. Aggregate duplicate ``(pre, post)`` rows by summing synapse counts.
5. Write unsigned counts as CSR plus compact neuron metadata.

Sign and incoming-sum normalization are **not** baked into the stored weights;
they are applied at simulation load time from ``LIFConfig``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.parquet as pq
from scipy import sparse

from fly_pilot.brain.config import (
    EXPECTED_EDGE_COUNT,
    EXPECTED_NEURON_COUNT,
    EXPECTED_WEIGHT_ROWS,
    INHIBITORY_TRANSMITTERS,
    MALECNS_DATASET,
    MALECNS_VERSION,
    PREPARED_SCHEMA,
    SOURCE_FILES,
    default_data_dir,
    prepared_dir,
    raw_dir,
)

WEIGHT_BATCH_ROWS = 2_000_000


class DataError(RuntimeError):
    """Authoritative MaleCNS data is missing, corrupt, or unexpected."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "fly-pilot-malecns/0.3")
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if existing and response.status == 206 else "wb"
        if mode == "wb" and partial.exists():
            existing = 0
        with partial.open(mode) as handle:
            copied = existing
            last_report = copied
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                copied += len(block)
                if copied - last_report >= 50 << 20:
                    print(f"  {copied / (1 << 20):.0f} MiB", flush=True)
                    last_report = copied
    partial.replace(destination)


def ensure_source_file(key: str, data_dir: Path) -> Path:
    spec = SOURCE_FILES[key]
    path = raw_dir(data_dir) / str(spec["filename"])
    expected_sha = str(spec["sha256"])
    expected_size = int(spec["size_bytes"])
    if path.exists():
        size = path.stat().st_size
        if size == expected_size and sha256_file(path) == expected_sha:
            return path
        print(f"Replacing unexpected {path.name} ({size} bytes)", flush=True)
        path.unlink()
    _download(str(spec["url"]), path)
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_sha:
        raise DataError(
            f"{path.name} failed validation: size={size} (expected {expected_size}), "
            f"sha256={digest} (expected {expected_sha}). "
            "This is not a substitute dataset; check the Janelia GCS source."
        )
    return path


def transmitter_sign(labels: np.ndarray) -> np.ndarray:
    """Map consensus transmitter strings to +1 / −1.

    Matching is case-insensitive exact membership in ``INHIBITORY_TRANSMITTERS``.
    Unknown, missing, and neuromodulatory labels are +1. This is an
    approximation, not synapse-level physiology.
    """
    sign = np.ones(len(labels), dtype=np.int8)
    for i, label in enumerate(labels):
        if str(label).strip().lower() in INHIBITORY_TRANSMITTERS:
            sign[i] = -1
    return sign


def _string_column(table: pa.Table, names: Iterable[str], fill: str = "") -> np.ndarray:
    columns = set(table.column_names)
    for name in names:
        if name in columns:
            filled = pc.fill_null(table[name], fill)
            return np.asarray(filled.to_pylist(), dtype=object)
    return np.full(table.num_rows, fill, dtype=object)


def _first_by_key(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first occurrence of each key (stable)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    first = np.ones(len(sorted_keys), dtype=bool)
    if len(sorted_keys) > 1:
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep = order[first]
    keep.sort()
    return keys[keep], values[keep]


def retained_neuron_ids(annotations: pa.Table) -> np.ndarray:
    body_col = "bodyId" if "bodyId" in annotations.column_names else "body"
    keep = pc.not_equal(pc.fill_null(annotations["superclass"], ""), "")
    bodies = annotations[body_col].combine_chunks()
    filtered = bodies.filter(keep)
    ids = np.unique(filtered.to_numpy())
    ids.sort()
    return ids.astype(np.int64, copy=False)


def _annotation_row_for_ids(annotations: pa.Table, ids: np.ndarray) -> pa.Table:
    body_col = "bodyId" if "bodyId" in annotations.column_names else "body"
    bodies = annotations[body_col].to_numpy()
    _, first_idx = _first_by_key(bodies, np.arange(len(bodies)))
    unique_bodies = bodies[first_idx]
    order = np.argsort(unique_bodies, kind="stable")
    unique_bodies = unique_bodies[order]
    first_idx = first_idx[order]
    positions = np.searchsorted(unique_bodies, ids)
    if np.any(positions >= len(unique_bodies)) or not np.array_equal(unique_bodies[positions], ids):
        missing = int(np.sum(unique_bodies[np.minimum(positions, len(unique_bodies) - 1)] != ids))
        raise DataError(f"{missing} retained body ids missing from annotations")
    return annotations.take(pa.array(first_idx[positions]))


def _transmitter_for_ids(transmitters: pa.Table, ids: np.ndarray) -> np.ndarray:
    body_col = "body" if "body" in transmitters.column_names else "bodyId"
    bodies = transmitters[body_col].to_numpy()
    labels = _string_column(transmitters, ("consensus_nt", "consensusNt"), fill="unclear")
    labels = np.array([str(x).strip().lower() if x not in (None, "") else "unclear" for x in labels], dtype=object)
    uniq_bodies, uniq_labels = _first_by_key(bodies, labels)
    order = np.argsort(uniq_bodies, kind="stable")
    uniq_bodies = uniq_bodies[order]
    uniq_labels = uniq_labels[order]
    pos = np.searchsorted(uniq_bodies, ids)
    found = pos < len(uniq_bodies)
    found[found] = uniq_bodies[pos[found]] == ids[found]
    out = np.full(len(ids), "unclear", dtype=object)
    out[found] = uniq_labels[pos[found]]
    return out


def map_body_ids(sorted_ids: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map unordered body ids onto dense indices of ``sorted_ids``."""
    index = np.searchsorted(sorted_ids, query)
    valid = index < len(sorted_ids)
    # Avoid reading past the end: only compare where searchsorted landed inside.
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        valid[valid_idx] = sorted_ids[index[valid_idx]] == query[valid_idx]
    return index.astype(np.int32, copy=False), valid


def aggregate_edges(
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    n_neurons: int,
) -> sparse.csr_matrix:
    """Build a CSR of summed synapse counts. Autapses are kept."""
    matrix = sparse.csr_matrix(
        (weight.astype(np.float32, copy=False), (post.astype(np.int32, copy=False), pre.astype(np.int32, copy=False))),
        shape=(n_neurons, n_neurons),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _iter_weight_batches(path: Path) -> Iterator[pa.RecordBatch]:
    table = feather.read_table(
        path,
        columns=["body_pre", "body_post", "weight"],
        memory_map=True,
    )
    if table.num_rows != EXPECTED_WEIGHT_ROWS:
        raise DataError(
            f"unexpected weight-table row count: {table.num_rows} (expected {EXPECTED_WEIGHT_ROWS})"
        )
    yield from table.to_batches(max_chunksize=WEIGHT_BATCH_ROWS)


def build_from_tables(
    annotations: pa.Table,
    transmitters: pa.Table,
    weight_batches: Iterable[pa.RecordBatch],
    *,
    expected_neurons: int | None = EXPECTED_NEURON_COUNT,
    expected_edges: int | None = EXPECTED_EDGE_COUNT,
) -> tuple[np.ndarray, pa.Table, sparse.csr_matrix, dict]:
    """Convert in-memory MaleCNS tables into a compact connectome.

    ``expected_*`` may be None in unit tests that use synthetic tables.
    """
    ids = retained_neuron_ids(annotations)
    if expected_neurons is not None and len(ids) != expected_neurons:
        raise DataError(f"unexpected retained-neuron count: {len(ids)} (expected {expected_neurons})")

    neuron_rows = _annotation_row_for_ids(annotations, ids)
    nt_labels = _transmitter_for_ids(transmitters, ids)
    signs = transmitter_sign(nt_labels)

    pre_parts: list[np.ndarray] = []
    post_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    raw_rows = 0
    mapped_rows = 0
    synapse_sum_mapped = 0

    for batch_number, batch in enumerate(weight_batches, 1):
        raw_rows += batch.num_rows
        pre_id = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post_id = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        w = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        pre, pre_ok = map_body_ids(ids, pre_id)
        post, post_ok = map_body_ids(ids, post_id)
        valid = pre_ok & post_ok
        if valid.any():
            pre_i = pre[valid]
            post_i = post[valid]
            values = w[valid]
            pre_parts.append(pre_i)
            post_parts.append(post_i)
            weight_parts.append(values)
            mapped_rows += int(valid.sum())
            synapse_sum_mapped += int(values.sum())
        if batch_number % 10 == 0:
            print(f"  mapped {min(batch_number * WEIGHT_BATCH_ROWS, raw_rows):,} source rows", flush=True)

    if not pre_parts:
        raise DataError("no edges mapped onto retained neurons")

    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    weights = np.concatenate(weight_parts)
    del pre_parts, post_parts, weight_parts

    autapses = int(np.sum(pre == post))
    matrix = aggregate_edges(pre, post, weights, len(ids))
    del pre, post, weights

    if expected_edges is not None and matrix.nnz != expected_edges:
        raise DataError(
            f"unexpected directed-edge count after aggregation: {matrix.nnz} "
            f"(expected {expected_edges})"
        )

    side = _string_column(neuron_rows, ("somaSide",), fill="")
    root_side = _string_column(neuron_rows, ("rootSide",), fill="")
    side = np.array([s if s else r for s, r in zip(side, root_side)], dtype=object)

    meta = pa.table(
        {
            "body_id": ids,
            "type": _string_column(neuron_rows, ("type",), fill=""),
            "flywire_type": _string_column(neuron_rows, ("flywireType", "flywire_type"), fill=""),
            "instance": _string_column(neuron_rows, ("instance",), fill=""),
            "superclass": _string_column(neuron_rows, ("superclass",), fill=""),
            "class": _string_column(neuron_rows, ("class",), fill=""),
            "subclass": _string_column(neuron_rows, ("subclass",), fill=""),
            "side": side,
            "consensus_nt": nt_labels.astype(object),
            "sign": signs,
        }
    )

    stats = {
        "n_neurons": int(len(ids)),
        "n_edges": int(matrix.nnz),
        "source_weight_rows": int(raw_rows),
        "mapped_weight_rows": int(mapped_rows),
        "autapse_rows": autapses,
        "synapse_count_sum": int(matrix.data.sum()),
        "inhibitory_neurons": int(np.sum(signs < 0)),
        "excitatory_neurons": int(np.sum(signs > 0)),
    }
    return ids, meta, matrix, stats


def prepared_is_current(data_dir: Path) -> bool:
    manifest_path = prepared_dir(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("schema") != PREPARED_SCHEMA:
        return False
    sources = manifest.get("source_sha256") or {}
    for key, spec in SOURCE_FILES.items():
        if sources.get(key) != spec["sha256"]:
            return False
    weights = prepared_dir(data_dir) / "weights.npz"
    neurons = prepared_dir(data_dir) / "neurons.parquet"
    return weights.exists() and neurons.exists()


def write_prepared(
    data_dir: Path,
    meta: pa.Table,
    matrix: sparse.csr_matrix,
    stats: dict,
    source_sha256: dict[str, str],
) -> Path:
    dest = prepared_dir(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(dest / "weights.npz", matrix, compressed=False)
    pq.write_table(meta, dest / "neurons.parquet", compression="zstd")
    manifest = {
        "schema": PREPARED_SCHEMA,
        "dataset": MALECNS_DATASET,
        "version": MALECNS_VERSION,
        "selection": (
            "all neurons with a non-empty superclass annotation, including tbc; "
            "no status filter; no synapse-count threshold"
        ),
        "weight_meaning": (
            "CSR entries are aggregated synapse counts at the dataset default "
            "PSD confidence (minconf 0.5). Duplicate (pre, post) rows were summed. "
            "Sign and incoming-sum normalization are applied at simulation load, "
            "not stored in weights.npz."
        ),
        "inhibitory_rule": (
            "GABA, glutamate, and histamine → −1; other/missing/unclear → +1 "
            "(Dale-style point-neuron approximation, not validated physiology)"
        ),
        "sources": {key: {"url": spec["url"], "filename": spec["filename"]} for key, spec in SOURCE_FILES.items()},
        "source_sha256": source_sha256,
        "source_md5": {key: spec["md5"] for key, spec in SOURCE_FILES.items()},
        "license": "CC BY 4.0 (MaleCNS data); FlyPilot code does not relicense the connectome",
        "attribution": (
            "MaleCNS v1.0, FlyEM Project Team (HHMI Janelia), Drosophila Connectomics "
            "Group (Cambridge / MRC LMB), Google Research. Berg et al., Cell (2026)."
        ),
        **stats,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dest / "manifest.json"


def prepare(data_dir: Path | None = None, *, force: bool = False) -> dict:
    data_dir = data_dir or default_data_dir()
    if prepared_is_current(data_dir) and not force:
        manifest = json.loads((prepared_dir(data_dir) / "manifest.json").read_text())
        print(
            f"Prepared MaleCNS already present: {manifest['n_neurons']:,} neurons, "
            f"{manifest['n_edges']:,} edges in {prepared_dir(data_dir)}",
            flush=True,
        )
        return manifest

    started = time.perf_counter()
    paths = {key: ensure_source_file(key, data_dir) for key in SOURCE_FILES}
    source_sha256 = {key: str(SOURCE_FILES[key]["sha256"]) for key in SOURCE_FILES}

    print("Reading neuron annotations and transmitters…", flush=True)
    annotations = feather.read_table(paths["annotations"])
    transmitters = feather.read_table(paths["transmitters"], columns=["body", "consensus_nt"])
    print("Mapping weighted edges (this walks the 152M-row table in batches)…", flush=True)
    _ids, meta, matrix, stats = build_from_tables(
        annotations,
        transmitters,
        _iter_weight_batches(paths["weights"]),
    )
    del annotations, transmitters
    write_prepared(data_dir, meta, matrix, stats, source_sha256)
    elapsed = time.perf_counter() - started
    stats["prepare_seconds"] = round(elapsed, 3)
    print(
        f"Prepared {stats['n_neurons']:,} neurons and {stats['n_edges']:,} directed edges "
        f"in {elapsed:.1f}s → {prepared_dir(data_dir)}",
        flush=True,
    )
    return json.loads((prepared_dir(data_dir) / "manifest.json").read_text())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download official MaleCNS v1.0 flat-connectome tables and write a compact "
            "local representation. Does not connect the network to the aircraft."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild even if prepared data is current")
    args = parser.parse_args(argv)
    prepare(args.data_dir, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
