# Visual embodiment (Milestone 4)

EXPERT + FLY OBSERVING: `ExpertLandingController` flies the JSBSim Cessna.
MaleCNS watches a rendered approach through a modeled fly-eye encoder.
**MaleCNS does not write any JSBSim inceptor.** No aircraft decoder is trained.

## REAL / MODELED / CONVENTIONAL

Keep these layers separate in code, HUD copy, and writing.

| Layer | What | Status |
|---|---|---|
| **REAL** | JSBSim `c172p` flight dynamics | Authoritative physics |
| **REAL** | Three.js landing scene (runway, ground, sky, Cessna mesh) | Human visualization |
| **REAL** | MaleCNS v1.0 measured neuron-to-neuron connectivity and annotations | 166,700 neurons, 25,582,938 edges |
| **REAL** | Actual R1–R6 / descending-neuron **identities** from those annotations | Body IDs, types, sides |
| **MODELED** | Python cubemap renderer used as the canonical fly view | Geometry + optic flow, not photorealism |
| **MODELED** | Approximate R1–R6 spatial layout (no optic-column Excel in the prepared tables) | Body-id rank → spherical grid |
| **MODELED** | Luminance / temporal contrast / spatial contrast / green drive | Engineering encoder |
| **MODELED** | Simplified LIF dynamics (`LIFConfig`) | Not measured physiology |
| **CONVENTIONAL** | `ExpertLandingController` | Classical cascaded PID; not a fly |

The airplane is still **not** controlled by MaleCNS.

## Research (architectural reference only)

[ornata/fly](https://github.com/ornata/fly) (Fly64) was inspected and is **not
copied** (no license file). Ideas reimplemented from
`docs/research-notes.md`:

- Six-face cubemap separate from the human framebuffer
- Wide-field coverage inspired by NeuroMechFly v2 (~270° horizontal, ~17°
  overlap, ±72° elevation)
- Seven-sample ~2° acceptance cone
- Drive mixing luminance, |ΔL|, and a green term
- Neural time owned by a 20 ms clock, not the display FPS
- Compact replay of sensory input without the original renderer

Not taken: mmap game bridge, Mario DN decoder, hand-mapped motor outputs,
dashboard 3D brain, 128² native cubemap faces (we use 32²).

### MaleCNS visual input actually present

Prepared annotations (`superclass` nonempty, including `tbc`):

| Population | How selected | Typical n |
|---|---|---:|
| R1–R6 photoreceptors | `type` / `flywireType` `R1-R6` / `R1-6` | **3,377** |
| L1 / L2 / L3 | `type` | recorded as debug rates |
| visual_projection | `superclass` | recorded as debug rates |
| descending_neuron | `superclass=descending_neuron` | **1,314** candidates |

The optic-column Excel workbook is **not** in `data/malecns/prepared/`. There
are no measured ommatidial pointing vectors in our tables. Spatial mapping is
therefore an approximation (below).

## Assumptions

1. Canonical sensory input is the **Python** cubemap, not the Three.js preview.
   The browser fly-eye canvases are a human visualization of similar geometry.
2. Each retained R1–R6 cell gets one visual direction. Cells are split by
   annotated side (`somaSide` / `rootSide` / `_L`/`_R` instance). Unknown sides
   are assigned even `body_id` → left, odd → right (deterministic leftover).
3. Within each eye, cells are sorted by `body_id` and stretched over a regular
   spherical grid covering:
   - left: azimuth −135° … +8.5°
   - right: azimuth −8.5° … +135°
   - elevation ±72°
   Azimuth 0 is the nose; positive is right.
4. RGB is an engineering stand-in. Drosophila photoreceptors are not Rec.709.
   There is no UV channel.
5. Temporal contrast uses the previous **neural** frame (50 Hz), not display
   frames.
6. Currents stored and injected are **float16-canonical**: the encoder
   round-trips through `float16` before the LIF step so live observation and
   parquet replay are the same numbers.
7. We do not claim that MaleCNS will steer. Left/right retinal differences are
   a spatial check, not a taxis result.
8. The connectome is not pruned for speed. CSC event propagation is sparse
   evaluation of the full annotated graph.

## Visual cameras

### Canonical (Python, drives MaleCNS)

Six faces, 32×32 RGB each, packed into a 96×64 atlas:

`forward | right | back`  
`left    | up    | down`

Eye origin: 2.4 m forward, 1.2 m up of the JSBSim reference point, in the
aircraft visual frame (+X right, +Y up, −Z forward). The Cessna mesh is **not**
drawn (the observer looks *from* the airplane).

Scene content: sky gradient + directional sun bias, fogged ground with a
40 m world-space checker, runway pavement, shoulder, centerline dashes,
threshold bars. Geometry and optic flow matter; scenery is not photoreal.

### Preview (Three.js, humans only)

Two off-screen 48×32 cameras, FOV 140°, yaw ±63.25° in the aircraft frame.
The Cessna mesh is hidden for that pass. HUD/telemetry live in DOM, not in
the Three.js scene, so they never appear in the fly image. Chase/cockpit
cameras remain separate.

These previews are **not** serialized into the dataset.

## Fly-eye encoder

For each mapped receptor:

1. Sample the cubemap with a 7-tap cone (~2°).
2. Rec.709 luminance `L`.
3. Temporal term `|L − L_prev|` (zero on the first frame after reset).
4. Spatial contrast `|center − mean(ring)|`.
5. Green term `max(G − 0.5(R+B), 0)`.
6. `drive = clip(0.45 L + 1.6 |ΔL| + 0.35 contrast + 0.25 green, 0, 1)`
7. `current = float16(drive * 0.62)` → float32 for the LIF.

Receptors looking in different directions receive different samples. They are
never all given the same current unless the atlas is spatially uniform.

## Transport

```
JSBSim pose
  → Python cubemap (canonical)
    → RetinalStimulusFrame (currents + luminance + SHA-256)
      → CurrentStimulation into actual R1–R6 indices
        → MaleCNS LIF @ 50 Hz
```

The browser does **not** send pixels to Python. Headless recording and replay
need no Three.js. The compact canonical object is `RetinalStimulusFrame`
(`retinal-v1`): float16 current blob, float16 luminance blob, current SHA-256,
optional atlas SHA-256.

## Timing

Browser `requestAnimationFrame` must not define neural time.

| Clock | Rate | Owner |
|---|---:|---|
| JSBSim physics | 120 Hz | FDM `dt = 1/120` |
| Visual observation / retinal encode | 50 Hz | `SimScheduler` on **sim** time |
| MaleCNS LIF | 50 Hz (`dt = 20 ms`) | same instants as vision |
| Neural / retinal logging | 50 Hz | same instants |
| WebSocket state broadcast | ~30 Hz | display only |
| Three.js fly-eye blit | display FPS, throttled ~12 Hz | preview only |

Neural step `k` fires the first time `sim_time_s >= k * 0.02`, including `k = 0`
at reset. The same stored retinal sequence replayed into a reset network must
reproduce the recorded spike-checksum chain regardless of frontend FPS.

## EXPERT + FLY OBSERVING

Mode name in protocol/HUD: `expert_observing` / **EXPERT + FLY OBSERVING**.

- Control authority is always `ExpertLandingController` (`controller.name == "expert"`).
- `ObservingMaleCNS` has **no** `act()` and `controls_aircraft is False`.
- `LandingSandbox.step_once` calls `self.controller.act()` only.
- `set_controller("fly_control"|"malecns"|…)` raises. No `MaleCNSController`.
- HUD banner: **FLY OBSERVING — NOT CONTROLLING**.

## Descending-neuron features

`DescendingNeuronFeatureExtractor` selects every neuron with
`superclass=descending_neuron` (1,314 in the prepared graph). It does **not**
hand-pick four cells or map them to aileron/elevator/rudder/throttle.

Per neural step it stores:

- per-neuron spike rates over 5-step (100 ms) and 13-step (260 ms) windows
- left/right mean rates from annotations
- compact HUD summary (mean / L / R / max Hz)

Selected visual-pathway rates (R1–R6, L1–L3, visual_projection, DNp01, DNa02,
DNg13, DNg100) are logged for debugging only.

## Dataset (`observing-expert-v1`)

```bash
python -m fly_pilot.record_observing --episodes 3 --seed 0 \
  --output data/observing/expert_observing.parquet
```

Parquet plus `.meta.json` sidecar. Each neural row includes:

- episode id, seed, sim timestamp, neural timestep
- runway-relative aircraft state, altitude, airspeed, VS, attitude, rates
- expert aileron / elevator / rudder / throttle
- retinal currents + luminance (float16) and SHA-256 / atlas checksum
- descending-neuron rate vector (float16) and selected visual rates
- total spike count, outgoing-edge workload, spike checksum

The full 166,700-neuron boolean raster is **not** stored. Sidecar lists
photoreceptor and DN body IDs so later decoder work can align columns.

## Replay

```bash
python -m fly_pilot.brain.replay_episode data/observing/expert_observing.parquet
```

Loads stored retinal currents, resets MaleCNS with the recorded seed/LIF
config, steps without Three.js or JSBSim, and requires the spike-checksum
sequence to match.

## UI

When `expert_observing` is active the HUD shows the observing banner, left/right
48×32 fly-eye canvases, retinal L/R luminance, temporal contrast, MaleCNS
spikes/s, DN mean and L/R rates, and R1–R6 / DN counts. Expert phase readouts
remain visible. Integrity copy still says the fly is not controlling.

## Validation

```bash
python -m fly_pilot.validate_vision
```

| Experiment | Expectation |
|---|---|
| A static pose | Retinal currents settle; same seed reproduces checksums |
| B approach motion | Retinal hashes change over the approach |
| C left vs right yaw | Atlas and retinal spatial pattern differ. No steering claim. |
| D replay | Recorded currents replay to matching checksums |

## Commands

```bash
python -m fly_pilot.brain.benchmark --steps 200 --json-out artifacts/malecns-benchmark-regimes.json
python -m fly_pilot.validate_vision --json-out artifacts/vision-experiments.json
python -m fly_pilot.record_observing --episodes 3 --seed 0
python -m fly_pilot.brain.replay_episode data/observing/expert_observing.parquet
```

Synthetic (no MaleCNS download) recordings use `--synthetic` and are what the
unit tests exercise.
