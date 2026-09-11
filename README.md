# FlyPilot

Use a real JSBSim Cessna 172 as the physics core of a browser landing sandbox, then later put the public MaleCNS fruit-fly connectome in the control loop.

**Current milestone: 3 — standalone MaleCNS-derived spiking-network simulation.**
The connectome is **not** in the Cessna control loop.
`ExpertLandingController` is still a classical PID autopilot — not biological
computation. There is no retina and no aircraft decoder yet.

## What currently works

- Python runs a real **JSBSim** `c172p` (Cessna 172P) flight dynamics model.
- Each episode starts ~3.2 km before a sea-level runway, ~250 m AGL, heading-aligned, ~70 KIAS.
- A browser client sends aileron / elevator / rudder / throttle over WebSocket.
- JSBSim integrates the aircraft; Python streams authoritative state back.
- Three.js renders runway, terrain, a Cessna mesh, chase/cockpit cameras, and a telemetry HUD from that state.
- Reset, successful-touchdown, crash, out-of-bounds, and failed-approach detection.
- Only `ManualController` and `ExpertLandingController` are implemented. The latter is a **conventional autopilot**, not a fly brain. MaleCNS controllers are not stubbed.
- Standalone MaleCNS v1.0 LIF simulator: prepare / query / demo / benchmark without JSBSim or the browser.

JSBSim is authoritative. The browser does not integrate aircraft motion.

## Run locally

```bash
bash scripts/install.sh
bash scripts/start.sh
```

Then open http://127.0.0.1:5173/

| Process | Command | Port |
|---|---|---|
| Backend (JSBSim + WebSocket) | `bash scripts/start-backend.sh` | `8765` |
| Frontend (Vite + Three.js) | `bash scripts/start-frontend.sh` | `5173` |

The Vite dev server proxies `/ws` to the backend.

### Tests

```bash
bash scripts/test.sh
```

Or separately:

```bash
source .venv/bin/activate
PYTHONPATH=backend python -m pytest
cd frontend && npm run typecheck
python -m fly_pilot.evaluate --episodes 100 --seed 0
python -m fly_pilot.record_expert --episodes 20 --output data/expert/demonstrations.parquet
python -m fly_pilot.brain.prepare
python -m fly_pilot.brain.demo
python -m fly_pilot.brain.benchmark
```

## Controls

| Input | Action |
|---|---|
| W / ↑ | Nose down (push) |
| S / ↓ | Nose up (pull) |
| A / ← | Roll left |
| D / → | Roll right |
| Q / E | Rudder |
| Shift / Ctrl | Throttle up / down |
| R | Reset episode |
| C | Chase ↔ cockpit camera |
| P | Pause / resume |
| HUD sliders and buttons | Same inceptors, plus reset/camera |
| MANUAL / EXPERT | Human vs conventional autoland (not MaleCNS) |

Elevator uses **pilot stick convention**: positive is back-stick / nose-up. The backend negates this for JSBSim's `fcs/elevator-cmd-norm`.

## Cloud agents

`.cursor/environment.json` installs dependencies with `scripts/install.sh` and starts both processes with `scripts/start.sh`.

Install is idempotent and does **not** start servers. Start launches JSBSim first, then Vite in the foreground.

See `AGENTS.md` for architecture, pitfalls, and scientific-integrity rules.

## Scientific integrity

These are different experiments and must stay labeled as such:

1. Fixed MaleCNS connectome + trained output decoder
2. MaleCNS with actual synaptic plasticity
3. An external learned controller (no connectome in the loop)

4. **ExpertLandingController** — classical cascaded PID; solvability baseline only

5. **Standalone MaleCNS LIF (Milestone 3)** — measured MaleCNS wiring, modeled
   firing dynamics, **not** in the aircraft loop

Milestone 3 adds (5) beside the Milestone 2 sandbox. The airplane is still not
fly-controlled. See `docs/malecns.md`.

## Layout

```
backend/fly_pilot/   JSBSim sandbox, controllers, WebSocket server, MaleCNS LIF
frontend/          Three.js / Vite client
tests/             pytest (includes live JSBSim checks and brain tests)
docs/              architecture, expert controller, MaleCNS notes
scripts/           install / start / test
data/malecns/      gitignored cache (run `python -m fly_pilot.brain.prepare`)
.cursor/           Cloud environment
```
