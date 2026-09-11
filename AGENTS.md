# FlyPilot agent notes

## Current milestone

**Milestone 4 — visual embodiment: the fly watches the expert land.**

Working loops:

```
keyboard/HUD → WebSocket → ManualController → JSBSim c172p → state → Three.js
ExpertLandingController (classical PID) → JSBSim c172p → state → Three.js
ExpertLandingController → JSBSim → Python cubemap → R1–R6 encoder → MaleCNS
```

MaleCNS output has **zero** effect on JSBSim. Mode name: `expert_observing`
(**EXPERT + FLY OBSERVING**), never “FLY CONTROL”.

This is a MaleCNS-derived spiking-network simulation using the measured
connectome. LIF dynamics and the retina are modeling choices. Do not call it
an exact fly brain, a living fly, validated neural dynamics, or proof of fly
cognition.

Do **not** implement `MaleCNSController`, `TrainedMaleCNSController`, or an
aircraft decoder in this milestone. Do not let MaleCNS activity reach
`apply_controls`.

`ExpertLandingController` remains ordinary autopilot code and must never be
presented as biological computation.

See `docs/vision.md` for the retinal path, scheduler, dataset, and replay.

## Architecture

- `backend/fly_pilot/aircraft.py` — JSBSim `FGFDMExec`, model `c172p`. Cranks the Lycoming on reset.
- `backend/fly_pilot/controllers/` — `Controller.observe()` / `act() -> AircraftControls`.
  - `ManualController` — browser inceptors
  - `ExpertLandingController` — cascaded PID autoland (classical, labelled as such)
- `backend/fly_pilot/brain/` — MaleCNS load / LIF / vision / observing (no `act()`)
- `backend/fly_pilot/brain/scheduler.py` — sim-time clocks (physics 120 Hz, vision/neural 50 Hz)
- `backend/fly_pilot/guidance.py` — runway-relative glideslope / heading helpers
- `backend/fly_pilot/initial_conditions.py` — seeded modest approach randomization
- `backend/fly_pilot/sandbox.py` — realtime step, reset, pause, controller switch
- `backend/fly_pilot/episode.py` — land / crash / OOB / failed approach
- `backend/fly_pilot/evaluate.py` — headless ≥100-episode report
- `backend/fly_pilot/record_expert.py` — Parquet demonstration dump
- `backend/fly_pilot/record_observing.py` — fly-observing-expert Parquet + sidecar
- `backend/fly_pilot/server.py` — WebSocket on port 8765
- `frontend/` — Vite + Three.js. Aircraft transform comes from JSBSim ENU + Euler angles.
- `frontend/src/flyEye.ts` — HUD preview cameras only; Python cubemap is canonical.

JSBSim remains the only physics integrator. Three.js must not dead-reckon a parallel airplane.
Browser FPS must not determine MaleCNS time.

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
python -m fly_pilot.brain.benchmark --steps 200 --json-out artifacts/malecns-benchmark-regimes.json
python -m fly_pilot.record_observing --episodes 3 --seed 0 --output data/observing/expert_observing.parquet
python -m fly_pilot.brain.replay_episode data/observing/expert_observing.parquet
python -m fly_pilot.validate_vision
```

Frontend talks to `ws(s)://<host>/ws`; Vite proxies that to `ws://127.0.0.1:8765`.

MaleCNS cache: `data/malecns/` (gitignored raw/prepared). Override with
`FLYPILOT_MALECNS_DIR`. Prepare is idempotent (SHA-256). Do not commit the
connectome. See `docs/malecns.md`.

Observing recordings: `data/observing/` (gitignored parquet). See `docs/vision.md`.

## Testing

Backend tests in `tests/` hit a real JSBSim `c172p` (elevator/aileron response, reset, telemetry, expert closed-loop landings). Do not mock the FDM for those.

Brain unit tests use tiny synthetic graphs. Full-network smoke tests run only
when `data/malecns/prepared/manifest.json` exists (`python -m fly_pilot.brain.prepare` first).

Vision / scheduler / replay tests are in `tests/test_vision.py` (synthetic graph).

Frontend check: `cd frontend && npm run typecheck`.

Visual check (required after UI/physics changes):

1. Start the app.
2. Confirm HUD shows `JSBSim connected` and live IAS/ALT.
3. MANUAL: move elevator/aileron; the mesh and telemetry must change.
4. EXPERT: aircraft follows the runway, descends via JSBSim (not teleport), flares, touches down. HUD shows phase / GS / XTK. Integrity copy says conventional autopilot, not MaleCNS.
5. EXPERT + FLY OBSERVING: same expert landing, banner **FLY OBSERVING — NOT CONTROLLING**, left/right fly-eye canvases update, spikes/s and DN rates move, inceptors still match the expert. Switching away from this mode stops observing.
6. Control surfaces follow JSBSim `fcs/*-pos-norm`.
7. Reset starts a new episode. Switching MANUAL ↔ EXPERT ↔ EXPERT+FLY OBSERVING works.
8. Browser console should stay free of app errors.

Do not add a fake MaleCNS controller option.

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
5. **MaleCNS LIF — measured wiring, modeled dynamics**
6. **Milestone 4 observing — modeled retina → MaleCNS; expert still flies; no decoder**

HUD copy must keep saying MaleCNS does not write inceptors, and that EXPERT is a conventional autopilot.

Do not copy Fly64 source. It has no license file. Use `docs/research-notes.md`, `docs/malecns.md`, and `docs/vision.md` as the implementation reference.

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
- CSS `display: grid` on `.expert` / `.observing` overrides the `hidden` attribute unless `[hidden] { display: none; }`.
- MaleCNS weight Feather is ~1 GiB and ~152 M rows. Stream it in batches; do not `to_pandas()` the edge table. Validate SHA-256. Keep unsigned counts on disk; apply sign/normalization at load.
- Dense CSR `W @ spikes` on 25.6 M edges was ~40 steps/s on this 4-vCPU VM. CSC event propagation (outgoing edges of spiking cells only) was ~190 steps/s without dropping edges.
- Status filters on MaleCNS annotations drop photoreceptors. Keep every nonempty superclass, including `tbc`.
- `flywireType` uses `R1-6`; `type` uses `R1-R6`. Population `--cell-type` matches both.
- An infinite uniform ground plane is translation-invariant. The fly-view shader uses fog, a world-space checker, sun bias, and runway markings so approach motion and yaw change the retina.
- Store retinal currents as float16 **and** inject the float16-round-tripped values into the LIF so replay without the renderer matches live checksums.
- Do not let the Three.js fly-eye blit become the recorded sensory source. Python cubemap is canonical.

## Later controllers (do not stub)

When adding the connectome to the aircraft, add real classes only:

- `MaleCNSController`
- `TrainedMaleCNSController`

Each must actually compute inceptors. Empty “TODO fly the plane” classes are forbidden. Keep `ExpertLandingController` labelled as conventional.
