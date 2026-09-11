# FlyPilot agent notes

## Current milestone

**Milestone 1 — browser-controlled Cessna 172 landing sandbox.**

Working loop:

```
keyboard/HUD → WebSocket → ManualController → JSBSim c172p → state → Three.js
```

Do **not** implement MaleCNS, training, or fake fly-brain placeholders until Milestone 1 stays green.

## Architecture

- `backend/fly_pilot/aircraft.py` — JSBSim `FGFDMExec`, model `c172p`.
- `backend/fly_pilot/controllers/` — `Controller.observe()` / `act() -> AircraftControls`. Only `ManualController` exists.
- `backend/fly_pilot/sandbox.py` — realtime step, reset, pause.
- `backend/fly_pilot/episode.py` — land / crash / OOB / failed approach.
- `backend/fly_pilot/server.py` — WebSocket on port 8765.
- `frontend/` — Vite + Three.js. Aircraft transform comes from JSBSim ENU + Euler angles.

JSBSim remains the only physics integrator. Three.js must not dead-reckon a parallel airplane.

## Commands

```bash
bash scripts/install.sh     # venv, pip, npm; no servers
bash scripts/start.sh       # backend :8765, frontend :5173
bash scripts/test.sh        # pytest + tsc
```

Frontend talks to `ws(s)://<host>/ws`; Vite proxies that to `ws://127.0.0.1:8765`.

## Testing

Backend tests in `tests/` hit a real JSBSim `c172p` (elevator/aileron response, reset, telemetry). Do not mock the FDM for those.

Frontend check: `cd frontend && npm run typecheck`.

Visual check (required after UI/physics changes):

1. Start the app.
2. Confirm HUD shows `JSBSim connected` and live IAS/ALT.
3. Move elevator/aileron; the mesh and telemetry must change.
4. Reset must snap the airplane back onto short final.
5. Browser console should stay free of app errors.

## Cloud-specific

- Install belongs in `scripts/install.sh` via `.cursor/environment.json`.
- Never put the Vite/JSBSim loop in `install`.
- `start` must leave a foreground process (Vite). Backend is backgrounded by `scripts/start.sh`.
- Bind `0.0.0.0`. Set Vite `allowedHosts: true`.
- Python package lives in `backend/`; `PYTHONPATH=backend` or `pip install -e backend`.
- This image may lack `python3-venv`; install script installs it when possible.

## Scientific integrity

Never silently replace MaleCNS with a conventional network and still call the project fly-controlled.

Distinguish, in code comments, HUD copy, and docs:

1. Fixed MaleCNS + trained decoder
2. MaleCNS with synaptic plasticity
3. External learned controller

Milestone 1 HUD copy must keep saying MaleCNS is not in the loop.

Do not copy Fly64 source. It has no license file. Use `docs/research-notes.md` as the implementation reference.

## Pitfalls discovered while building

- `ic/lat-geod-deg` vs `position/lat-gc-deg` differ by ~0.18° at 37°N. Use **geodetic** latitude on both IC and readout (`ic/lat-geod-deg`, `position/lat-geod-deg`).
- Approach `do_trim(0)` failed (`udot` not trimmable) with a descending gamma. Milestone 1 uses ICs + `ic/alpha-deg` instead of trim.
- `propulsion/set-running = -1` raises on c172p; use `propulsion/engine/set-running = 1`.
- JSBSim `fcs/elevator-cmd-norm` positive is nose-down. Browser elevator is stick-back positive; `aircraft.py` applies a minus sign.
- `reset_to_initial_conditions` works, but a full `run_ic()` after rewriting ICs is the reset used here. Re-apply mixture/flaps/engine after reset.
- `query_property_catalog()` returns a string, not a list; do not iterate it as properties.
- Default JSBSim `dt` is 1/120 s. Visual stream is ~30 Hz; FDM still steps at 120 Hz.
- Vite must proxy `/ws`; the browser should not hard-code `localhost:8765` (breaks Cloud previews).
- After land/crash the sandbox pauses so the airplane is not driven into the mesh. Reset clears `paused`.
- Visual frame is X=east, Y=up, Z=south so heading 0 looks down −Z (north). Pose mapping is in `applyJsbsimPose`.

## Later controllers (do not stub)

When Milestone 1 is solid, add real classes only:

- `ExpertLandingController`
- `MaleCNSController`
- `TrainedMaleCNSController`

Each must actually compute inceptors. Empty “TODO fly the plane” classes are forbidden.
