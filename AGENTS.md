# FlyPilot agent notes

## Current milestone

**Milestone 5 — FLY CONTROL: fixed MaleCNS + trained temporal decoder.**

Working loops:

```
keyboard/HUD → WebSocket → ManualController → JSBSim c172p → state → Three.js
ExpertLandingController (classical PID) → JSBSim c172p → state → Three.js
ExpertLandingController → JSBSim → Python cubemap → R1–R6 encoder → MaleCNS
  (EXPERT + FLY OBSERVING; MaleCNS does not write inceptors)
aircraft pose → Python cubemap → R1–R6 → MaleCNS LIF → DN 100/260 ms rates
  → trained GRU decoder → AircraftControls → JSBSim
  (FLY CONTROL)
```

`TrainedMaleCNSController` is an **external decoder**. MaleCNS synapses are
frozen. Do not call this biological synaptic learning, an exact fly brain, or
proof of fly cognition.

`ExpertLandingController` remains ordinary autopilot code. It must never be
blended into FLY CONTROL and must never be presented as biological computation.

Decoder input is **only** MaleCNS descending-neuron windowed spike rates.
Aircraft telemetry must not enter the decoder.

See `docs/decoder.md` and `docs/vision.md`.

## Architecture

- `backend/fly_pilot/aircraft.py` — JSBSim `FGFDMExec`, model `c172p`. Cranks the Lycoming on reset.
- `backend/fly_pilot/controllers/` — `Controller.observe()` / `act() -> AircraftControls`.
  - `ManualController` — browser inceptors
  - `ExpertLandingController` — cascaded PID autoland (classical, labelled as such)
  - `TrainedMaleCNSController` — frozen MaleCNS LIF + trained GRU decoder
- `backend/fly_pilot/brain/` — MaleCNS load / LIF / vision / observing / decoder
- `backend/fly_pilot/brain/decoder.py` — causal GRU, checkpoint, DN-only input
- `backend/fly_pilot/brain/scheduler.py` — sim-time clocks (physics 120 Hz, vision/neural 50 Hz)
- `backend/fly_pilot/guidance.py` — runway-relative glideslope / heading helpers
- `backend/fly_pilot/initial_conditions.py` — seeded spawn randomization (`DECODER_SPAWN` is wider)
- `backend/fly_pilot/disturbance.py` — executed-command pulses during expert recording
- `backend/fly_pilot/sandbox.py` — realtime step, reset, pause, controller switch
- `backend/fly_pilot/episode.py` — land / crash / OOB / failed approach
- `backend/fly_pilot/evaluate.py` — headless expert report
- `backend/fly_pilot/evaluate_decoder.py` — offline GRU vs Ridge vs mean
- `backend/fly_pilot/evaluate_fly.py` — closed-loop FLY CONTROL report + videos
- `backend/fly_pilot/record_expert.py` — Parquet demonstration dump
- `backend/fly_pilot/record_observing.py` — fly-observing-expert Parquet + sidecar (retinal blobs)
- `backend/fly_pilot/record_decoder.py` — compact DN-rate / expert-action dataset
- `backend/fly_pilot/train_decoder.py` — episode-split GRU training
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

python -m fly_pilot.record_decoder --successes 50 --seed 1000 --output data/decoder/expert_dn_controls.parquet
python -m fly_pilot.train_decoder --data data/decoder/expert_dn_controls.parquet --output artifacts/decoder/best.pt
python -m fly_pilot.evaluate_decoder --checkpoint artifacts/decoder/best.pt --data data/decoder/expert_dn_controls.parquet
python -m fly_pilot.evaluate_fly --episodes 20 --seed 2000 --json artifacts/decoder/fly-eval.json
```

Frontend talks to `ws(s)://<host>/ws`; Vite proxies that to `ws://127.0.0.1:8765`.

MaleCNS cache: `data/malecns/` (gitignored raw/prepared). Override with
`FLYPILOT_MALECNS_DIR`. Prepare is idempotent (SHA-256). Do not commit the
connectome. See `docs/malecns.md`.

Decoder checkpoint: `artifacts/decoder/best.pt` (gitignored). Override with
`FLYPILOT_DECODER_PATH`.

Observing recordings: `data/observing/` (gitignored parquet). Compact decoder
training set: `data/decoder/`. See `docs/vision.md` and `docs/decoder.md`.

## Testing

Backend tests in `tests/` hit a real JSBSim `c172p` (elevator/aileron response, reset, telemetry, expert closed-loop landings). Do not mock the FDM for those.

Brain unit tests use tiny synthetic graphs. Full-network smoke tests run only
when `data/malecns/prepared/manifest.json` exists (`python -m fly_pilot.brain.prepare` first).

Vision / scheduler / replay tests are in `tests/test_vision.py` (synthetic graph).
Decoder / FLY CONTROL integrity tests are in `tests/test_decoder.py`.

Frontend check: `cd frontend && npm run typecheck`.

Visual check (required after UI/physics changes):

1. Start the app.
2. Confirm HUD shows `JSBSim connected` and live IAS/ALT.
3. MANUAL: move elevator/aileron; the mesh and telemetry must change.
4. EXPERT: aircraft follows the runway, descends via JSBSim (not teleport), flares, touches down. HUD shows phase / GS / XTK. Integrity copy says conventional autopilot, not MaleCNS.
5. EXPERT + FLY OBSERVING: same expert landing, banner **FLY OBSERVING — NOT CONTROLLING**, left/right fly-eye canvases update, spikes/s and DN rates move, inceptors still match the expert. Switching away from this mode stops observing.
6. FLY CONTROL: banner **FLY CONTROL** / **FIXED MALECNS + TRAINED TEMPORAL DECODER**. Decoder aileron/elevator/rudder/throttle move. Inceptors match the decoder, not the expert. Switching away stops fly authority.
7. Control surfaces follow JSBSim `fcs/*-pos-norm`.
8. Reset starts a new episode. Switching MANUAL ↔ EXPERT ↔ EXPERT+FLY OBSERVING ↔ FLY CONTROL works.
9. Browser console should stay free of app errors.

Do not add a fake untrained `MaleCNSController`. `malecns` without a trained decoder stays rejected.

## Cloud-specific

- Install belongs in `scripts/install.sh` via `.cursor/environment.json`.
- Never put the Vite/JSBSim loop in `install`.
- `start` must leave a foreground process (Vite). Backend is backgrounded by `scripts/start.sh`.
- Bind `0.0.0.0`. Set Vite `allowedHosts: true`.
- Python package lives in `backend/`; `PYTHONPATH=backend` or `pip install -e backend`.
- This image may lack `python3-venv`; install script installs it when possible.
- MaleCNS download is **not** part of `install.sh` (≈1.1 GiB). Run `python -m fly_pilot.brain.prepare` once per environment; reuse `data/malecns/`.
- Decoder training is **not** part of `install.sh`. Checkpoint path: `FLYPILOT_DECODER_PATH`.

## Scientific integrity

Never silently replace MaleCNS with a conventional network and still call the project fly-controlled.

Distinguish, in code comments, HUD copy, and docs:

1. Fixed MaleCNS + trained decoder  ← **this milestone (FLY CONTROL)**
2. MaleCNS with synaptic plasticity
3. External learned controller
4. **ExpertLandingController — classical autopilot; not (1), (2), or (3) as a fly model**
5. **MaleCNS LIF — measured wiring, modeled dynamics**
6. **Milestone 4 observing — modeled retina → MaleCNS; expert still flies; no decoder**
7. **Milestone 5 FLY CONTROL — (1): frozen MaleCNS + external temporal decoder; not (2)**

HUD copy must keep saying EXPERT is a conventional autopilot, and that FLY
CONTROL is a trained decoder rather than biological learning.

Do not copy Fly64 source. It has no license file. Use `docs/research-notes.md`, `docs/malecns.md`, `docs/vision.md`, and `docs/decoder.md` as the implementation reference.

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
- CSS `display: grid` on `.expert` / `.observing` / `.fly-control` overrides the `hidden` attribute unless `[hidden] { display: none; }`.
- MaleCNS weight Feather is ~1 GiB and ~152 M rows. Stream it in batches; do not `to_pandas()` the edge table. Validate SHA-256. Keep unsigned counts on disk; apply sign/normalization at load.
- Dense CSR `W @ spikes` on 25.6 M edges was ~40 steps/s on this 4-vCPU VM. CSC event propagation (outgoing edges of spiking cells only) was ~190 steps/s without dropping edges.
- Status filters on MaleCNS annotations drop photoreceptors. Keep every nonempty superclass, including `tbc`.
- `flywireType` uses `R1-6`; `type` uses `R1-R6`. Population `--cell-type` matches both.
- An infinite uniform ground plane is translation-invariant. The fly-view shader uses fog, a world-space checker, sun bias, and runway markings so approach motion and yaw change the retina.
- Store retinal currents as float16 **and** inject the float16-round-tripped values into the LIF so replay without the renderer matches live checksums.
- Do not let the Three.js fly-eye blit become the recorded sensory source. Python cubemap is canonical.
- Descending-neuron rates must divide by the **available** buffer length during warmup, not the configured window, or early rates are biased low.
- Train/val/test splits must be by **complete episode**, never shuffled rows.
- FLY CONTROL neural step runs on the **current** pose **before** `act()`, so the decoder is causal with the image the fly currently sees.
- Do not fall back to ExpertLandingController if the fly diverges.

## Later controllers

- `TrainedMaleCNSController` is implemented (external decoder).
- `MaleCNSController` (untrained / hand-mapped) is still absent and must not be stubbed.
Keep `ExpertLandingController` labelled as conventional.
