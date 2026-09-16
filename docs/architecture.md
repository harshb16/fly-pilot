# Architecture

## Goal

Eventually:

```
Three.js cockpit camera
  → fruit-fly retinal encoder
  → MaleCNS connectome simulation
  → descending-neuron activity
  → trainable decoder
  → aileron / elevator / rudder / throttle
  → JSBSim Cessna 172
  → aircraft state
  → Three.js
```

Milestone 1 implements the plant, transport, visualization, and a human controller.
Milestone 2 adds `ExpertLandingController`, a **conventional classical autopilot**
used as a solvability baseline and expert-data generator.
Milestone 3 adds a **standalone** MaleCNS-derived LIF simulator.
Milestone 4 adds **visual observation**: a fly-view renderer and R1–R6 encoder
drive MaleCNS while the expert still flies.
Milestone 5 adds **FLY CONTROL**: a trained causal GRU reads descending-neuron
rates and writes JSBSim inceptors. MaleCNS synapses stay frozen. See
`docs/decoder.md`.

## Runtime processes

1. **`fly_pilot.server`** (Python, port 8765)
   - Owns one `LandingSandbox`.
   - Steps JSBSim in wall-clock time (FDM dt = 1/120 s, catch-up capped).
   - Accepts JSON commands: `controls`, `reset`, `pause`, `resume`, `set_controller`
     (`manual` | `expert` | `expert_observing` | `fly_control`).
   - Broadcasts `hello` once per client and `state` at ~30 Hz.
   - In `expert_observing` and `fly_control`, steps MaleCNS on the 50 Hz sim-time
     scheduler **before** `act()` so vision is causal with the current pose.
     `fly_control` sends decoder outputs to JSBSim. `expert_observing` does not.
2. **Vite / Three.js** (port 5173)
   - Proxies `/ws` to the backend.
   - Sends manual inceptors.
   - Renders JSBSim pose; does not integrate flight dynamics.

## Backend modules

| Module | Role |
|---|---|
| `state.py` | `AircraftControls`, `AircraftObservation`, episode enum |
| `controllers/base.py` | `Controller.observe` / `act` |
| `controllers/manual.py` | Stores browser inceptors |
| `controllers/expert.py` | Conventional cascaded-PID autoland (not MaleCNS) |
| `controllers/trained.py` | FLY CONTROL: frozen MaleCNS + GRU decoder |
| `brain/decoder.py` | Causal GRU, scaler, portable checkpoint |
| `controllers/pid.py` | Discrete PID with anti-windup |
| `guidance.py` | Glideslope / heading-error geometry |
| `initial_conditions.py` | Seeded modest approach randomization |
| `aircraft.py` | JSBSim `c172p` load, IC, engine crank, step, telemetry |
| `geodesy.py` | WGS84 metres/deg ENU around the runway origin |
| `runway.py` | Threshold, heading 090, 1200×30 m, approach spawn |
| `episode.py` | Land / crash / OOB / failed-approach rules |
| `evaluate.py` | Headless expert evaluation |
| `record_expert.py` | Parquet expert demonstrations |
| `brain/` | MaleCNS prepare / LIF / vision encoder / observing / replay |
| `record_observing.py` | Parquet fly-observing-expert dataset (retinal blobs) |
| `record_decoder.py` | Compact DN-rate / expert-action training set |
| `train_decoder.py` | Episode-split causal GRU training |
| `evaluate_decoder.py` | Offline GRU vs Ridge vs mean |
| `evaluate_fly.py` | Closed-loop FLY CONTROL landings |
| `sandbox.py` | Glue: controller → FDM → optional MaleCNS observe → episode |
| `protocol.py` | JSON schema |
| `server.py` | `websockets` server + sim loop |

### Controller contract

```python
class Controller:
    def reset(self) -> None: ...
    def observe(self, observation: AircraftObservation) -> None: ...
    def act(self) -> AircraftControls: ...
```

`AircraftControls` is always:

- aileron ∈ [-1, 1]  (positive = roll right)
- elevator ∈ [-1, 1] (positive = stick back / nose up)
- rudder ∈ [-1, 1]   (positive = yaw right)
- throttle ∈ [0, 1]

JSBSim elevator sign conversion happens only in `aircraft.py`.

`LandingSandbox.set_controller("manual"|"expert"|"expert_observing"|"fly_control")`.
`malecns` without a trained decoder is still refused. `expert_observing` still
instantiates `ExpertLandingController` as the sole `act()` source.
`fly_control` instantiates `TrainedMaleCNSController`; the expert is absent.

Clocks (simulated time, never `requestAnimationFrame`):

| Clock | Hz |
|---|---:|
| JSBSim | 120 |
| Vision / retinal / MaleCNS / log | 50 |
| State broadcast | ~30 |

### Coordinate frames

- **JSBSim:** WGS84 geodetic lat/lon, MSL/AGL feet, Euler angles, 120 Hz.
- **Local ENU:** origin = runway threshold. East, north, up metres.
- **Runway frame:** along-track (landing direction) and right-track.
- **Three.js:** `x = east`, `y = up`, `z = -north`. Heading 0 looks along −Z.

### Episode origin

Default approach (inside the 2–4 km / 200–400 m window):

- 3200 m before threshold
- 250 m AGL
- 70 KIAS
- runway heading 090
- flight-path −4.5°, alpha +4°
- flaps 0.4, mixture 0.9, throttle 0.42
- engine forced running

## Frontend

Vanilla TypeScript. No React.

- `simClient.ts` — WebSocket + reconnect
- `input.ts` — keys + HUD sliders
- `aircraftMesh.ts` — low-poly high-wing Cessna; `applyJsbsimPose`
- `runwayMesh.ts` — pavement, markings, chevrons, hills
- `scene.ts` — lights, fog, chase / cockpit cameras (human view)
- `flyEye.ts` — off-screen wide-FOV previews; not the canonical MaleCNS input
- `hud.ts` — telemetry, MANUAL / EXPERT / EXPERT+FLY OBSERVING / FLY CONTROL, integrity copy

## What is explicitly out of Milestone 5

- Pretending the expert autopilot is a fly
- Claiming biological synaptic learning from the GRU
- Untrained / hand-mapped `MaleCNSController`
- Photoreal scenery / exact compound-eye optics
- React
- Wind

See `docs/malecns.md`, `docs/vision.md`, and `docs/decoder.md`. The connectome
must stay a real component if the project is described as fly-controlled.
