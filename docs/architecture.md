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
drive MaleCNS while the expert still flies. There is still **no** aircraft
decoder and MaleCNS still does **not** write inceptors. See `docs/vision.md`.

## Runtime processes

1. **`fly_pilot.server`** (Python, port 8765)
   - Owns one `LandingSandbox`.
   - Steps JSBSim in wall-clock time (FDM dt = 1/120 s, catch-up capped).
   - Accepts JSON commands: `controls`, `reset`, `pause`, `resume`, `set_controller`
     (`manual` | `expert` | `expert_observing`).
   - Broadcasts `hello` once per client and `state` at ~30 Hz.
   - In `expert_observing`, steps MaleCNS on the 50 Hz sim-time scheduler after
     each physics tick that crosses a neural instant. Neural output is telemetry
     only.
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
| `record_observing.py` | Parquet fly-observing-expert dataset (not a decoder) |
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

`LandingSandbox.set_controller("manual"|"expert"|"expert_observing")` is the
only legal switch. Names that would imply fly control (`malecns`,
`fly_control`, …) are refused. `expert_observing` still instantiates
`ExpertLandingController` as the sole `act()` source; `ObservingMaleCNS` has
no `act()`. `ExpertLandingController.telemetry()` is labelled
`kind: conventional_autopilot`.

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
- `hud.ts` — telemetry, MANUAL / EXPERT / EXPERT+FLY OBSERVING, integrity copy

## What is explicitly out of Milestone 4

- MaleCNS writing JSBSim inceptors (`MaleCNSController` / decoder)
- Pretending the expert autopilot is a fly
- Claiming biological steering from left/right retinal differences
- Photoreal scenery / exact compound-eye optics
- React
- Wind

See `docs/malecns.md` and `docs/vision.md`. The connectome must stay a real
component if a later milestone is described as fly-controlled.
