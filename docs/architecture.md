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

Milestone 1 implements only the plant, transport, visualization, and a human controller.

## Runtime processes

1. **`fly_pilot.server`** (Python, port 8765)
   - Owns one `LandingSandbox`.
   - Steps JSBSim in wall-clock time (FDM dt = 1/120 s, catch-up capped).
   - Accepts JSON commands: `controls`, `reset`, `pause`, `resume`.
   - Broadcasts `hello` once per client and `state` at ~30 Hz.
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
| `aircraft.py` | JSBSim `c172p` load, IC, engine, step, telemetry |
| `geodesy.py` | WGS84 metres/deg ENU around the runway origin |
| `runway.py` | Threshold, heading 090, 1200×30 m, approach spawn |
| `episode.py` | Land / crash / OOB / failed-approach rules |
| `sandbox.py` | Glue: controller → FDM → episode → snapshot |
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
- `scene.ts` — lights, fog, chase / cockpit cameras
- `hud.ts` — telemetry and integrity copy

## What is explicitly out of Milestone 1

- MaleCNS download / sparse LIF / retina
- Training (CMA-ES, RL, PyTorch)
- Expert autoland
- Photoreal scenery
- React

Those belong in later milestones, with the connectome remaining a real component if the project is described as fly-controlled.
