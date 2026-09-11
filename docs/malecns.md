# MaleCNS-derived spiking-network simulation (Milestone 3)

This document describes a **standalone** MaleCNS simulator inside FlyPilot.

It is acceptable to say:

> MaleCNS-derived spiking-network simulation using the measured connectome.

It is **not** an exact fly brain, a living-fly simulation, biologically
validated neural dynamics, or proof of fly cognition.

- **Anatomy:** real MaleCNS v1.0 neuron-to-neuron connectivity and annotations.
- **Dynamics:** a simplified leaky-integrate-and-fire (LIF) model, a modeling
  choice inspired by current MaleCNS demos (Fly64), not measured physiology.

Milestone 4 **observes** the Cessna through a modeled retina. It still does
**not** let MaleCNS write JSBSim inceptors and does **not** train an aircraft
decoder. See `docs/vision.md`.

## Commands

```bash
python -m fly_pilot.brain.prepare      # download/validate/compact (idempotent)
python -m fly_pilot.brain.query --list-superclasses
python -m fly_pilot.brain.query --cell-type DNp01 --show-ids
python -m fly_pilot.brain.demo         # baseline → stim → recovery + ablation
python -m fly_pilot.brain.benchmark    # multi-regime timings on this machine
python -m fly_pilot.brain.replay_episode data/observing/expert_observing.parquet
python -m fly_pilot.record_observing --episodes 3 --seed 0
python -m fly_pilot.validate_vision
```

Override the cache directory with `FLYPILOT_MALECNS_DIR`. Default:
`<repo>/data/malecns/` (`raw/` and `prepared/` are gitignored).

Re-running `prepare` does **not** re-download if SHA-256 matches, and does
not rebuild if `prepared/manifest.json` is current.

## Dataset / provenance

| Field | Value |
|---|---|
| Dataset | MaleCNS **v1.0** (`male-cns:v1.0`) |
| Portal | https://male-cns.janelia.org/download/ |
| License | **CC BY 4.0** |
| Producers | FlyEM (HHMI Janelia), Drosophila Connectomics Group (Cambridge / MRC LMB), Google Research |
| Citation | Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, *Cell* 189(18):5504–5526.e15 (2026). https://doi.org/10.1016/j.cell.2026.08.015 |
| Representation | Official **flat-connectome** Feather tables (neuron-to-neuron), **not** the 100M+ raw synapse list |

### Files downloaded (authoritative GCS)

All three live under
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`.

| File | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `50a7718770c57220f160ba4f431ab89e` | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `3d842b12fe5c49eefade528d7dd24a1f` | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `f30e9dcca25cfd021bf1e7b3d975599e` | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

Hashes were measured on this Cursor Cloud VM after a successful HTTP GET and
are enforced by `python -m fly_pilot.brain.prepare`. Unexpected size or digest
fails the command; we do not fall back to synthetic graphs.

We did **not** use neuPrint Cypher as the primary pipeline. Anonymous neuPrint
(`POST /api/custom/custom` on `male-cns:v1.0`) is a documented alternate
access route (see fly-brain-minecraft provenance). The GCS Feather tables are
the smaller official aggregated representation.

Optic-column Excel, neuron meshes, SWC skeletons, and raw EM imagery are
intentionally not downloaded.

## What is loaded

Measured on this VM after `prepare`:

| Quantity | Value |
|---|---:|
| Retained neurons | **166,700** (non-empty `superclass`, including 94 `tbc`) |
| Directed edges | **25,582,938** |
| Source weight-table rows | 151,856,684 |
| Mapped rows (both ends retained) | 25,582,938 |
| Autapse rows (kept) | 101 |
| Sum of synapse-count weights | 124,177,616 |
| Inhibitory neurons (modeled −1) | 59,262 |
| Excitatory neurons (modeled +1) | 107,438 |

Annotation table has 211,577 bodies; 44,877 lack a superclass and are dropped.
That is a published-annotation filter, not a speed prune. Status-only filters
would drop photoreceptors and are **not** used.

### Synapses vs aggregated edges

The Feather `weight` column is already an **aggregated synapse count** for a
directed `(body_pre, body_post)` pair at the dataset default PSD confidence
(`minconf 0.5`). It is not a per-PSD table.

The 151.9 million source rows include connections whose endpoints are
unannotated fragments. After requiring **both** endpoints in the 166,700
superclass-annotated neurons, 25,582,938 rows remain. In this snapshot
`mapped_weight_rows == n_edges`, so there were no extra duplicate `(pre, post)`
pairs among retained neurons. Duplicate rows, if present, would be summed.
CSR stores one directed edge per unique pair; the data value is the synapse
count.

We do **not** apply a minimum-weight threshold (fly-brain-minecraft’s bundled
file keeps only `weight >= 5`, which is a size/speed cut). The complete
annotated graph stays in the simulation.

## Annotations available for queries

Stored per neuron: `body_id`, `type`, `flywire_type`, `instance`,
`superclass`, `class`, `subclass`, `side` (`somaSide`, else `rootSide`),
`consensus_nt`, modeled `sign`.

`python -m fly_pilot.brain.query` can filter any of those. `--cell-type`
matches `type` or `flywireType` (so `R1-6` and `R1-R6` both work). This is
not limited to descending neurons.

Superclass counts in the prepared graph (same as the annotation table after
the superclass filter):

| superclass | n |
|---|---:|
| ol_intrinsic | 89,403 |
| cb_intrinsic | 32,164 |
| vnc_intrinsic | 13,161 |
| visual_projection | 9,201 |
| vnc_sensory | 6,370 |
| ol_sensory | 6,098 |
| cb_sensory | 4,868 |
| ascending_neuron | 1,846 |
| descending_neuron | 1,314 |
| vnc_motor | 708 |
| … | (smaller classes in the query CLI) |

Examples: `R1-R6` / `R1-6` photoreceptors = 3,377; `L1` = 1,776; `DNp01`
(giant fiber) = 2 (`DNp01(GF)_R` / `_L`).

## Neural dynamics (modeling choice)

Each 20 ms step:

```
v ← exp(−dt / τ) · v
  + g · W · s          # recurrent, from anatomy
  + I_tonic
  + I_background       # seeded Bernoulli
  + I_stim             # optional external current
s ← [v ≥ θ]
v[s] ← v_reset
```

Default `LIFConfig` (not biophysical measurements):

| Parameter | Default | Role |
|---|---|---|
| `dt` | 0.020 s | 50 neural steps / simulated second |
| `τ` (`tau_m`) | 0.100 s | leak |
| `θ` | 1.0 | spike threshold |
| `v_reset` | 0.0 | reset |
| `I_tonic` | 0.180 | constant drive |
| `g` (`synaptic_gain`) | 1.50 | scales `W s` |
| background rate | 1.2 Hz | Bernoulli per neuron |
| background amplitude | 0.22 | on a background event |
| `normalize_incoming` | True | divide each postsynaptic row by Σ\|incoming\| |
| `seed` | 64 | NumPy Generator |

`W[post, pre] = sign[pre] · count[post, pre] / max(Σ_k |sign[k] count[post, k]|, 1)`
when normalization is on. **Counts** are stored unsigned in `weights.npz`.
Sign and normalization are applied at load time so the on-disk graph stays
anatomical.

### Excitatory / inhibitory treatment

`consensus_nt` from the official transmitter table, first row per body.

| Transmitter | Modeled sign |
|---|---|
| gaba, glutamate, histamine | −1 |
| acetylcholine, dopamine, octopamine, serotonin, unclear/missing | +1 |

This is the Shiu et al. 2024 / Fly64 **Dale-style point-neuron approximation**:
adult fly glutamate often acts through GluCl (inhibitory); histamine is the
photoreceptor transmitter. Receptor-dependent and neuromodulatory effects are
omitted. Tests inspect the sign table and a three-neuron graph where an
excitatory vs inhibitory presynaptic spike raises vs lowers postsynaptic
current. **Do not read this as synapse-resolved physiology.**

### Timestep justification

Fly64 steps its simplified model at 50 Hz. JSBSim here runs at 120 Hz, so a
later flight loop can take several FDM substeps per neural step. We use the
same 20 ms tick so the standalone simulator is already in the right
timebase. It is **not** a claim about Drosophila membrane kinetics.

## Disk and memory (this Cloud VM)

| Location | Size |
|---|---|
| `data/malecns/raw/` (3 Feather files) | 1.1 GiB |
| `data/malecns/prepared/weights.npz` | 196 MiB |
| `data/malecns/prepared/neurons.parquet` | 1.6 MiB |
| `data/malecns/` total | 1.3 GiB |

`prepare` on this VM: **13.6 s** from already-downloaded raw files (hash check
+ batch map of 152 M rows + CSR write).

Process RSS (`/proc/self/status` VmRSS), `python -m fly_pilot.brain.benchmark`:

| Point | RSS |
|---|---|
| before load | 86 MB |
| after `Connectome.load` | 445 MB |
| after LIF init (signed CSR + CSC) | 856 MB |

CSR/CSC numeric payload: 205 MB each (float32 data + int32 indices + indptr).

## Performance (measured, not theoretical)

Host: Cursor Cloud VM, Python 3.12.3, NumPy 2.5.3, 4 CPUs, 15 GiB RAM.

`python -m fly_pilot.brain.benchmark --steps 250 --warmup 30`

| Metric | Value |
|---|---|
| Connectome load | 0.39 s |
| LIF initialization | 0.74 s |
| Neural timesteps / wall-clock second | **190** |
| Simulated seconds / wall-clock second | **3.80** (dt = 0.020) |
| Mean spikes / step (this config) | 10,874 |

The first CSR-dense matvec implementation ran at ~40 steps/s. Profiling
showed the 25.6 M nonzero multiply was the cost; switching to **CSC event
propagation** (sum outgoing edges of the neurons that actually spiked) raised
throughput to 190 steps/s without removing any edge. Sparse evaluation ≠
graph pruning.

190 steps/s is 3.8× faster than the 50 Hz neural clock, enough headroom for
an eventual flight loop on this class of VM.

## Demo stimulation

`python -m fly_pilot.brain.demo` (seed 64, 50 baseline + 50 stim + 50 recovery):

- **Input population:** 3,377 `R1-R6` photoreceptors (synthetic current
  amplitude 1.2; not retinal pixels).
- **Watched:** `L1` (1,776), `descending_neuron` (1,314), `DNp01` (2).

| Phase | Global rate (Hz) | R1-R6 (Hz) | L1 (Hz) |
|---|---:|---:|---:|
| Baseline | 2.80 | 2.48 | 1.69 |
| Stimulation | 4.19 | **49.09** | 1.42 |
| Recovery | 3.26 | 2.37 | 1.81 |

Evidence:

1. Stimulation enters those cells: R1-R6 rate jumps to ~1 spike/step.
2. Activity uses the loaded graph: extra spikes **beyond** the stimulated
   pool ≈ **74,382** with recurrent weights vs ≈ **21,616** with recurrent
   disabled. Global spike count and checksums change. L1 rate **falls**
   during photoreceptor drive, consistent with this model’s histamine = −1
   approximation, **not** a biological validation.
3. Disabling recurrent connections changes the result (checksum
   `06a724b3…` vs `1cbd8680…`).

## Deterministic replay

Same prepared connectome + `LIFConfig` (including seed) + stimulation
sequence ⇒ identical boolean spike checksums.

The 150-step recurrent demo checksum
`06a724b3c8eeb0a4d3ee9a3ce45672c5ca4854bc57a1b5f71bab20d22b80521b`
matched across separate process invocations on this VM. Unit tests cover
seeded background, reset, and trace round-trip. `--trace` writes a compact
JSON checksum chain (not a full 166 k × T spike raster).

Replay is deterministic on the **same platform / NumPy / SciPy**. Float
reduction order is not promised across architectures.

## Comparison to Fly64 (`ornata/fly`)

Studied, **not copied** (that repository has no license file).

| Topic | Fly64 | FlyPilot Milestone 3 |
|---|---|---|
| Source tables | Same GCS Feather files | Same, hashes recorded |
| Neuron filter | 166,700 nonempty superclass | Same |
| Edges | 25,582,938, no weight cut | Same |
| On-disk W | Signed + row-normalized CSR | **Unsigned synapse counts**; sign/norm at load |
| Runtime | CSC event walk | CSC event walk |
| dt / LIF | 20 ms, τ=0.1, θ=1, tonic 0.18, g=1.5, Bernoulli 1.2 Hz × 0.22 | Same defaults, named `LIFConfig` |
| Vision | Cubemap → photoreceptors | Python 6×32 cubemap → R1–R6 (modeled layout) |
| Motor | Hand-written DN → Mario | **Not implemented** (DN features recorded only) |
| License | Unlicensed source; we do not vendor it | Original FlyPilot code; MaleCNS data remains CC BY 4.0 |

fly-brain-minecraft (MIT code, CC BY data) was used as a **provenance**
reference: anonymous neuPrint, `male-cns:v1.0` identifiers, `ConnectsTo.weight`
as synapse count. Their bundled graph thresholds at ≥ 5 synapses; we do not.

## Known limitations

- LIF parameters are engineered so that synthetic drive is visible, not fit
  to recordings.
- One sign per neuron (Dale’s law); mixed-transmitter and receptor logic
  are absent.
- Incoming-sum normalization is a stability trick, not anatomy.
- Background is i.i.d. Bernoulli, not measured spontaneous activity.
- Photoreceptor stimulation in the **demo** is injected current. Milestone 4
  observing mode uses the modeled retinal encoder instead.
- R1–R6 spatial layout is an approximate spherical grid (no optic-column Excel).
- Tonic current places many cells near threshold (~10 k spikes/step);
  that is a property of these defaults, not of a fly.
- Autapses are stored; a point neuron treats them as self-current.
- 44,877 unannotated fragments are omitted with their incident edges.
- No plasticity, neuromodulation, compartments, or delay.
- MaleCNS still does not control JSBSim. Observation is one-way.

## Implementation map

```
backend/fly_pilot/brain/
  data.py           download, hash, compact CSR + parquet
  connectome.py     load, body-id ↔ dense index, unsigned counts
  model.py          LIF + CSC event input
  populations.py    annotation queries
  stimulation.py    pulse + per-receptor CurrentStimulation
  telemetry.py      rates / checksums
  replay.py         spike-trace JSON (demo)
  replay_episode.py retinal-current replay of observing datasets
  vision/           cubemap, mapping, encoder, stimulus
  observing.py      EXPERT + FLY OBSERVING (no act())
  features.py       DN / visual-pathway rate extractors
  scheduler.py      sim-time clocks
  prepare.py / demo.py / benchmark.py / query.py
```

Tests: `tests/test_brain.py` (synthetic) and `tests/test_brain_network.py`
(full graph, skipped if unprepared). Existing sandbox tests are unchanged.
