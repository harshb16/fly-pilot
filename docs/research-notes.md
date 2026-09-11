# Research notes

Implementation reference for later agents. Fly64 source was inspected and is **not copied** (no license file in [ornata/fly](https://github.com/ornata/fly)). JSBSim is LGPL-2.1; we use the public Python module, not vendored JSBSim C++.

## Fly64 (ornata/fly)

Public experiment: MaleCNS v1.0 driving Super Mario 64 via a patched sm64ex build, a Python LIF network, and a dashboard on port 8765.

Documented loop:

```
Mario world → six-face cubemap → photoreceptors → LIF network → descending-neuron decoder → Mario controls
```

### MaleCNS dataset (as used by Fly64)

Source: https://male-cns.janelia.org/download/ (CC-BY; Berg et al.).

Fly64 cache files (do not download until a later milestone):

| File | URL pattern |
|---|---|
| annotations | `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather` |
| transmitters | `.../body-neurotransmitters-male-cns-v1.0.feather` |
| weights | `.../connectome-weights-male-cns-v1.0-minconf-0.5.feather` |
| optic columns | `https://github.com/flyconnectome/2025malecns` supplemental `optic-column-type-assignments-v1.0.xlsx` at commit `67767d2233657983993ff6c2be48e836a935863c` |

Counts they enforce:

- 166,700 cells with non-empty superclass (including 94 `tbc`)
- 25,582,938 directed weighted edges after keeping both endpoints
- Status-only filters would drop photoreceptors; they do not use that filter
- 6,006 R1–R8 photoreceptors
- 140,638 measured soma positions; 26,062 missing (excluded from the map only)

Weight rule: synapse count × presynaptic sign / incoming absolute-weight sum. GABA, glutamate, histamine treated as inhibitory; other/unknown excitatory. Approximation, not physiology.

Sparse format: float32 CSR on disk, CSC at runtime. Event propagation walks outgoing edges of spiking neurons only. That is sparse evaluation, not graph pruning.

### LIF dynamics (Fly64-specific; not measured)

Every 20 ms (50 Hz):

```
v ← exp(-dt/0.1)·v + 1.5·W·spikes + 0.180 + noise + retina
```

- τ = 0.1 s, threshold 1, reset 0
- Bernoulli background 1.2 Hz, amplitude 0.22, default seed 64
- Retinal current injected **only** into photoreceptors
- Tonic current and synaptic gain were calibrated so vision is visible downstream

### Retina / vision

- Native game renders six 128×128 views (front, right, back, left, up, down) into a 384×256 RGB atlas, 10 Hz
- Separate from the human framebuffer; no HUD, Mario mesh hidden
- Eyes cover ~270° horizontal, ±72° elevation, ~17° overlap (inspired by NeuroMechFly v2, not a copy of FlyGym)
- Seven-sample acceptance cone (~2°)
- Drive: 0.45·luminance + 1.6·|Δluminance| + 0.25·green-opponency
- RGB, not UV; angular registration of optic columns is engineered
- 2,628 cells have published columns; 3,242 connectivity-estimated; 136 within-eye proxy

### Descending-neuron readout (hand-written, not trained)

13-tick (~260 ms) window:

- DNg100 → forward
- DNa02 / DNg13 right-minus-left → steering
- DNp01 / DNp10 burst > 0.04 spikes/cell/tick → jump (800 ms cooldown)
- EMA 0.78/0.22, deadzone 8, stick ±70/127

Fly64 is explicit that this mapping is an engineering rule, not recovered fly motor physiology, and that there is **no training**.

### Environment bridge

Shared mmap file `runtime/fly64_bridge.bin`, 128-byte little-endian header + 294,912 RGB bytes, seqlocks, POSIX `CLOCK_MONOTONIC` heartbeat. Neural controls release after 250 ms stale. macOS Python `monotonic_ns()` is the **wrong clock**.

Replay: JSON index + NPZ chunks, every neural tick, can recompute spikes without SM64.

### Performance / debugging

- Dashboard at 127.0.0.1:8765: fisheye eyes, motor-pool rates, firing-rate map
- Dashboard publish drops 10→5 Hz if realtime factor < 0.95; neural ticks are never skipped
- Causality test: live vs frozen vs disconnected frames must yield different motor outputs when connected
- Synthetic 4,096-cell fixture via `--synthetic` / `--demo`

### Integrity lesson for FlyPilot

Keep three layers separate in code and writing:

1. Measured wiring (MaleCNS tables)
2. Engineered dynamics / retina / decoder
3. The external plant (SM64 there, JSBSim here)

Do not describe (2) or a trained decoder as “the fly”.

## JSBSim

Repo: https://github.com/JSBSim-Team/jsbsim — Python wheels on PyPI (`pip install jsbsim`).

Verified in this environment: **JSBSim 1.3.1**, model **`c172p`**, default `dt = 1/120`.

### Python pattern

```python
import jsbsim
fdm = jsbsim.FGFDMExec(None)   # packaged aircraft/engine data
fdm.set_debug_level(0)
fdm.load_model("c172p")
fdm["ic/lat-geod-deg"] = 37.0
fdm["ic/long-gc-deg"] = -122.0
fdm["ic/h-sl-ft"] = 820
fdm["ic/vc-kts"] = 70
fdm["ic/psi-true-deg"] = 90
fdm["ic/gamma-deg"] = -4.5
fdm["ic/alpha-deg"] = 4.0
fdm.run_ic()
fdm["propulsion/engine/set-running"] = 1
fdm["fcs/mixture-cmd-norm"] = 0.9
fdm["fcs/flap-cmd-norm"] = 0.4
fdm["fcs/aileron-cmd-norm"] = 0
fdm["fcs/elevator-cmd-norm"] = 0   # positive = nose down
fdm["fcs/rudder-cmd-norm"] = 0
fdm["fcs/throttle-cmd-norm"] = 0.42
fdm.run()
```

### Useful properties

| Property | Use |
|---|---|
| `position/lat-geod-deg` | Geodetic latitude |
| `position/long-gc-deg` | Longitude |
| `position/h-sl-ft`, `position/h-agl-ft` | Altitude |
| `velocities/vc-kts` | Calibrated airspeed |
| `velocities/vg-fps` | Ground speed |
| `velocities/h-dot-fps` | Climb rate (×60 → fpm) |
| `attitude/pitch-rad`, `attitude/roll-rad`, `attitude/psi-deg` | Euler |
| `aero/alpha-deg`, `aero/beta-deg` | Aero angles |
| `flight-path/gamma-deg` | Flight path |
| `fcs/*-cmd-norm`, `fcs/*-pos-norm` | Inceptors / surfaces |
| `gear/wow`, `gear/unit[n]/WOW` | Weight on wheels |
| `simulation/sim-time-sec` | Sim time |

`ic/lat-geod-deg = 37` does **not** equal `position/lat-gc-deg`. Always pair geodetic with geodetic.

### Cessna 172 models in-tree

`aircraft/c172p` (used), `c172r`, `c172x`. Reset XMLs exist (`reset00.xml` ground, `reset01.xml` airborne) but we set ICs from Python so the airplane spawns on a custom short final.

### Trim

`FGFDMExec.do_trim(0)` (longitudinal) failed on this descending approach (`udot` not trimmable). Do not depend on trim for Milestone 1. A later expert controller can try `do_trim` in level flight or a custom IC.

Engine: `propulsion/set-running = -1` errors (“non-existent engine”) on this model. Use `propulsion/engine/set-running`.

Control check (1 s, 70 kt): elevator cmd −0.4 (JSBSim) raised pitch ~12°; aileron +0.4 rolled right ~13°.

## Related Gym / JSBSim environments

Inspected as architecture references only (not copied):

- [Gor-Ren/gym-jsbsim](https://github.com/Gor-Ren/gym-jsbsim) — wraps `FGFDMExec`, dict of properties, `start_engines` via `propulsion/set-running` (does not work as-is on current c172p), throttle/mixture helpers, `sim_frequency_hz` often 60, FlightGear output optional.
- [sryu1/jsbgym](https://github.com/sryu1/jsbgym) — Gymnasium-era fork, same property catalogue.
- JSBSim README also cites gym-jsbsim for ML control.

Useful ideas taken (reimplemented):

- Property catalogue as named constants conceptually (we inlined the ones we need)
- Explicit engine start after `run_ic`
- Keep rendering out of the FDM process
- Terminate episodes on crash / out of bounds rather than running forever

Not taken: FlightGear socket output, OpenAI Gym API, reward shaping (no training yet).

## NeuroMechFly / embodied fly work (later)

- Wang-Chen et al. 2024, NeuroMechFly v2, *Nature Methods*. Wide-angle ommatidial views; Fly64 used the ~270° / 17° overlap as design inspiration only.
- MaleCNS paper / data portal for wiring.
- Fly64 dashboard and technical notes for how *not* to over-claim physiology.

Milestone 3 implemented the real Feather tables as a **standalone** LIF
simulator (`docs/malecns.md`). Retina, descending-neuron decoder, and
aircraft coupling remain later work. Keep sparse CSR/CSC; do not substitute
a dense MLP for the graph.
