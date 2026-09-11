# FlyPilot agent notes

## Current milestone

**Milestone 3 — standalone MaleCNS-derived spiking-network simulation.**

Working loops (unchanged from Milestone 2):

```
keyboard/HUD → WebSocket → ManualController → JSBSim c172p → state → Three.js
ExpertLandingController (classical PID) → JSBSim c172p → state → Three.js
```

New, **not** wired to the aircraft:

```
python -m fly_pilot.brain.prepare / demo / benchmark / query
```

This is a MaleCNS-derived spiking-network simulation using the measured
connectome. LIF dynamics are a modeling choice. Do not call it an exact fly
brain, a living fly, validated neural dynamics, or proof of fly cognition.

Do **not** implement `MaleCNSController`, `TrainedMaleCNSController`, retina,
or an aircraft decoder in this milestone.

`ExpertLandingController` remains ordinary autopilot code and must never be
presented as biological computation.

## Architecture

- `backend/fly_pilot/aircraft.py` — JSBSim `FGFDMExec`, model `c172p`. Cranks the Lycoming on reset.
- `backend/fly_pilot/controllers/` — `Controller.observe()` / `act() -> AircraftControls`.
  - `ManualController` — browser inceptors
  - `ExpertLandingController` — cascaded PID autoland (classical, labelled as such)
- `backend/fly_pilot/brain/` — standalone MaleCNS load / LIF / stim / replay (no JSBSim)
- `backend/fly_pilot/guidance.py` — runway-relative glideslope / heading helpers
- `backend/fly_pilot/initial_conditions.py` — seeded modest approach randomization
- `backend/fly_pilot/sandbox.py` — realtime step, reset, pause, controller switch
- `backend/fly_pilot/episode.py` — land / crash / OOB / failed approach
- `backend/fly_pilot/evaluate.py` — headless ≥100-episode report
- `backend/fly_pilot/record_expert.py` — Parquet demonstration dump
- `backend/fly_pilot/server.py` — WebSocket on port 8765
- `frontend/` — Vite + Three.js. Aircraft transform comes from JSBSim ENU + Euler angles.

JSBSim remains the only physics integrator. Three.js must not dead-reckon a parallel airplane.

## Commands

```bash
bash scripts/install.sh     # venv, pip, npm; no servers; does not download MaleCNS
bash scripts/start.sh       # backend :8765, frontend :5173
bash scripts/test.sh        # pytest + tsc
python -m fly_pilot.evaluate --episodes 100 --seed 0 --json artifacts/expert-eval.json
python -m fly_pilot.record_expert --episodes 50 --seed 0 --output data/expert/demonstrations.parquet

python -m fly_pilot.brain.prepare
python -m fly_pilot.brain.query --list-superclasses
python -m fly_pilot.brain.query --cell-type DNp01 --show-ids
python -m fly_pilot.brain.demo --trace artifacts/malecns-demo-trace.json
python -m fly_pilot.brain.benchmark
```

Frontend talks to `ws(s)://<host>/ws`; Vite proxies that to `ws://127.0.0.1:8765`.

MaleCNS cache: `data/malecns/` (gitignored raw/prepared). Override with
`FLYPILOT_MALECNS_DIR`. Prepare is idempotent (SHA-256). Do not commit the
connectome. See `docs/malecns.md`.

## Testing

Backend tests in `tests/` hit a real JSBSim `c172p` (elevator/aileron response, reset, telemetry, expert closed-loop landings). Do not mock the FDM for those.

Brain unit tests use tiny synthetic graphs. Full-network smoke tests run only
when `data/malecns/prepared/manifest.json` exists (`python -m fly_pilot.brain.prepare` first).

Frontend check: `cd frontend && npm run typecheck`.

Visual check (required after UI/physics changes):

1. Start the app.
2. Confirm HUD shows `JSBSim connected` and live IAS/ALT.
3. MANUAL: move elevator/aileron; the mesh and telemetry must change.
4. EXPERT: aircraft follows the runway, descends via JSBSim (not teleport), flares, touches down. HUD shows phase / GS / XTK. Integrity copy says conventional autopilot, not MaleCNS.
5. Control surfaces follow JSBSim `fcs/*-pos-norm`.
6. Reset starts a new episode. Switching MANUAL ↔ EXPERT works.
7. Browser console should stay free of app errors.

Milestone 3 does not change the HUD. Do not add a fake MaleCNS controller option.

## Cloud-specific

- Install belongs in `scripts/install.sh` via `.cursor/environment.json`.
- Never put the Vite/JSBSim loop in `install`.
- `start` must leave a foreground process (Vite). Backend is backgrounded by `scripts/start.sh`.
- Bind `0.0.0.0`. Set Vite `allowedHosts: true`.
- Python package lives in `backend/`; `PYTHONPATH=backend` or `pip install -e backend`.
- This image may lack `python3-venv`; install script installs it when possible.
- MaleCNS download is **not** part of `install.sh` (≈1.1 GiB). Run `python -m fly_pilot.brain.prepare` once per environment; reuse `data/malecns/`.

## Scientific integrity

Never silently replace MaleCNS with a conventional network and still call the project fly-controlled.

Distinguish, in code comments, HUD copy, and docs:

1. Fixed MaleCNS + trained decoder
2. MaleCNS with synaptic plasticity
3. External learned controller
4. **ExpertLandingController — classical autopilot; not (1), (2), or (3) as a fly model**
5. **Standalone MaleCNS LIF (Milestone 3) — measured wiring, modeled dynamics, not in the aircraft loop**

Milestone 2 HUD copy must keep saying MaleCNS is not in the loop, and that EXPERT is a conventional autopilot.

Do not copy Fly64 source. It has no license file. Use `docs/research-notes.md` and `docs/malecns.md` as the implementation reference.

## Pitfalls discovered while building

- `ic/lat-geod-deg` vs `position/lat-gc-deg` differ by ~0.18° at 37°N. Use **geodetic** latitude on both IC and readout (`ic/lat-geod-deg`, `position/lat-geod-deg`).
- Approach `do_trim(0)` failed (`udot` not trimmable) with a descending gamma. Milestone 1 uses ICs + `ic/alpha-deg` instead of trim.
- `propulsion/set-running = -1` raises on c172p; `propulsion/engine/set-running = 1` **also does not start** the piston engine on JSBSim 1.3.1. RPM stays 0 until you crank `propulsion/starter_cmd` with `propulsion/magneto_cmd = 3`. Then `run_ic()` again to restore the approach pose (engine stays running).
- JSBSim `fcs/elevator-cmd-norm` positive is nose-down. Browser elevator is stick-back positive; `aircraft.py` applies a minus sign.
- `reset_to_initial_conditions` works, but a full `run_ic()` after rewriting ICs is the reset used here. Re-apply mixture/flaps/engine after reset.
- `query_property_catalog()` returns a string, not a list; do not iterate it as properties.
- Default JSBSim `dt` is 1/120 s. Visual stream is ~30 Hz; FDM still steps at 120 Hz.
- Vite must proxy `/ws`; the browser should not hard-code `localhost:8765` (breaks Cloud previews).
- After land/crash the sandbox pauses so the airplane is not driven into the mesh. Reset clears `paused`.
- Visual frame is X=east, Y=up, Z=south so heading 0 looks down −Z (north). Pose mapping is in `applyJsbsimPose`.
- Do not iterate `self.clients` while awaiting sends; disconnect handlers mutate the set. Iterate `list(self.clients)`.
- Pause the FDM when no browser is connected, and `reset()` on the first new client. Otherwise a Cloud agent that starts the server then opens the UI minutes later finds the Cessna already past the runway.
- `run_ic()` restores ICs but does **not** zero `simulation/sim-time-sec` on JSBSim 1.3.1. Call `set_sim_time(0)` or episode timeouts accumulate across resets.
- Uncommanded C172 is spirally unstable. Zero aileron/rudder is not a wings-level hold.
- CSS `display: grid` on `.expert` overrides the `hidden` attribute unless `.expert[hidden] { display: none; }`.
- MaleCNS weight Feather is ~1 GiB and ~152 M rows. Stream it in batches; do not `to_pandas()` the edge table. Validate SHA-256. Keep unsigned counts on disk; apply sign/normalization at load.
- Dense CSR `W @ spikes` on 25.6 M edges was ~40 steps/s on this 4-vCPU VM. CSC event propagation (outgoing edges of spiking cells only) was ~190 steps/s without dropping edges.
- Status filters on MaleCNS annotations drop photoreceptors. Keep every nonempty superclass, including `tbc`.
- `flywireType` uses `R1-6`; `type` uses `R1-R6`. Population `--cell-type` matches both.

## Later controllers (do not stub)

When adding the connectome to the aircraft, add real classes only:

- `MaleCNSController`
- `TrainedMaleCNSController`

Each must actually compute inceptors. Empty “TODO fly the plane” classes are forbidden. Keep `ExpertLandingController` labelled as conventional.
