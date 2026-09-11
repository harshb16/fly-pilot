# FlyPilot

Use a real JSBSim Cessna 172 as the physics core of a browser landing sandbox, then later put the public MaleCNS fruit-fly connectome in the control loop.

**Current milestone: 1 — human-flown JSBSim landing sandbox.**
The MaleCNS model is **not** implemented and **not** in the loop.

## What currently works

- Python runs a real **JSBSim** `c172p` (Cessna 172P) flight dynamics model.
- Each episode starts ~3.2 km before a sea-level runway, ~250 m AGL, heading-aligned, ~70 KIAS.
- A browser client sends aileron / elevator / rudder / throttle over WebSocket.
- JSBSim integrates the aircraft; Python streams authoritative state back.
- Three.js renders runway, terrain, a Cessna mesh, chase/cockpit cameras, and a telemetry HUD from that state.
- Reset, successful-touchdown, crash, out-of-bounds, and failed-approach detection.
- Only `ManualController` is implemented. The controller interface is ready for later expert / MaleCNS controllers; those are not stubbed.

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

Milestone 1 is none of the above. It is a human flying a JSBSim Cessna so the later loop has a real plant.

## Layout

```
backend/fly_pilot/   JSBSim sandbox, controllers, WebSocket server
frontend/          Three.js / Vite client
tests/             pytest (includes live JSBSim checks)
docs/              architecture and research notes
scripts/           install / start / test
.cursor/           Cloud environment
```
