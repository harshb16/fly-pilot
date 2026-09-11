# ExpertLandingController

**This is a conventional classical autopilot. It is not MaleCNS, not a fruit-fly
connectome, and not biological computation.** It exists only as:

1. a solvability baseline for the JSBSim C172 landing environment,
2. an expert-demonstration generator for a later imitation / decoder stage,
3. a comparison benchmark for the eventual `MaleCNSController`.

Do not describe this controller as fly-controlled. Later classes
`MaleCNSController` and `TrainedMaleCNSController` are intentionally absent
from this milestone.

## Design

Cascaded PID / P loops, no learned weights:

| Loop | Outer | Inner | Effector |
|---|---|---|---|
| Lateral | centerline error → heading command | heading error → roll command → aileron | aileron + coordinated rudder |
| Vertical | glideslope height error → VS command → pitch | pitch error → elevator | elevator |
| Speed | IAS error | — | throttle |

Phases:

1. **stabilize** — first 2 s: capture wings-level / heading, 70 KIAS
2. **approach** — track 4.4° glideslope to an aiming point 120 m past threshold, 68 KIAS, runway centerline
3. **flare** — below 18 m AGL and within 150 m of the threshold: raise pitch, retard throttle
4. **touchdown** — below 3.5 m AGL: hold a slight nose-up attitude, idle
5. **rollout** — weight-on-wheels: idle, rudder to heading, aileron wings-level

Gains live in `ExpertLandingGains` and were tuned against live JSBSim `c172p`,
not against the Three.js mesh.

## Coordinate conventions

Unchanged from Milestone 1:

- JSBSim: geodetic lat/lon (`position/lat-geod-deg`, `ic/lat-geod-deg`), Euler angles, 120 Hz
- Local ENU: origin = runway threshold
- Runway frame: `along_m` positive in the landing direction (090), `right_m` positive to the right
- Pilot elevator: positive = stick back / nose up. `aircraft.py` negates this for `fcs/elevator-cmd-norm`
- Three.js: `x = east`, `y = up`, `z = -north`

Glideslope error: **positive means above the reference slope.**

Heading error: **positive means the aircraft must turn right** to the target.

## Important JSBSim properties

| Property | Role |
|---|---|
| `ic/lat-geod-deg`, `ic/long-gc-deg`, `ic/h-sl-ft` | spawn |
| `ic/vc-kts`, `ic/psi-true-deg`, `ic/phi-deg`, `ic/gamma-deg`, `ic/alpha-deg` | speed / attitude ICs |
| `fcs/aileron-cmd-norm`, `elevator-cmd-norm`, `rudder-cmd-norm`, `throttle-cmd-norm` | inceptors |
| `propulsion/starter_cmd`, `propulsion/magneto_cmd` | engine start |
| `propulsion/engine/engine-rpm`, `propulsion/engine/thrust-lbs` | confirm the Lycoming is alive |
| `velocities/p-rad_sec`, `q-rad_sec`, `r-rad_sec` | rate damping |
| `gear/wow`, `gear/unit[i]/WOW` | touchdown |

### Why the untrimmed aircraft sinks and spirals

Two genuine plant behaviours, not renderer bugs:

1. **Dead engine on Milestone 1 reset.** `propulsion/engine/set-running = 1` does **not** start the JSBSim 1.3.1 `c172p` Lycoming. RPM stayed 0 and thrust stayed 0, so throttle had no effect and the airframe glided. Milestone 2 cranks with starter + magnetos until RPM > 1000, then `run_ic()` again so the episode still begins on short final with a running engine.
2. **Spiral mode.** With aileron/rudder at zero the C172 is spirally unstable: a small bank grows, heading walks off, and the airplane leaves the centerline. The expert’s inner roll/yaw loops exist because of that, not to hide it.

The Three.js pose is still a direct map of JSBSim ENU + Euler angles. Control-surface meshes are driven by JSBSim `fcs/*-pos-norm`.

## Successful-landing criterion

Unchanged from `EpisodeMonitor` / `is_successful_touchdown`:

- first ground contact after being airborne
- over the 1200 × 30 m runway, with 8 m extra lateral tolerance
- sink rate ≤ 700 fpm (`-vertical_speed_fpm ≤ 700`)
- `|roll| ≤ 12°`, `|pitch| ≤ 12°`

Harder contacts (> 900 fpm), off-runway ground contact, or flying past the far end are crashes / failed approaches. The expert does not relax these thresholds.

## Randomization

`sample_spawn(runway, seed=N)` draws a modest final-approach IC:

- lateral offset ±80 m
- altitude ±30 m around 250 m AGL
- heading error ±8°
- airspeed ±5 kt around 70 kt
- pitch-related gamma/alpha and roll ±8°

No wind. The same integer seed is deterministic.

## Commands

```bash
# ≥100 headless episodes, no renderer
python -m fly_pilot.evaluate --episodes 100 --seed 0 --json artifacts/expert-eval.json

# Expert demonstrations (Parquet). Not MaleCNS training data.
python -m fly_pilot.record_expert --episodes 50 --seed 0 --output data/expert/demonstrations.parquet
```

## Evaluation results

Headless run (no renderer), 100 episodes, `--seed 0`, 2026-09-11, JSBSim 1.3.1 `c172p`:

```
episodes:           100
success rate:       100.0%  (100/100)
crash rate:         0.0%  (0)
failed approach:    0.0%  (0)

Successful touchdowns:
  mean sink:        491 fpm   (p90 491, worst 491)
  mean |xtk|:       1.28 m    (p90 1.29, worst 1.30)
  mean |hdg err|:   0.29 deg
  mean |roll|:      0.22 deg
  mean along:       371 m
  mean IAS:         56.7 kt
  mean time:        102.6 s
```

Re-run:

```bash
python -m fly_pilot.evaluate --episodes 100 --seed 0 --json artifacts/expert-eval.json
```

The near-identical touchdown sink/xtk across seeds is the controller capturing onto the same flare, not a loosened detector. The detector is still the Milestone 1 `is_successful_touchdown` rule (≤700 fpm, on runway, |roll|/|pitch| ≤ 12°). Elapsed time still varies with spawn (~98–110 s).

## Known limitations

- No wind, turbulence, or engine-out cases.
- Flare is a pitch/throttle schedule, not a full automatic landing system (no autobrakes; rollout uses rudder only).
- Touchdown sink is typically a few hundred fpm — within the 700 fpm cap, firmer than a greaser.
- Small steady-state centerline bias (~1 m) remains.
- Engine crank adds a few simulated seconds inside `Cessna172.reset()`; wall time is small.
- Expert data is (state, classical-action) pairs. A later milestone must pair **MaleCNS activity** with these actions; do not train a “fly” decoder on this file alone and call it biological.

## Integrating MaleCNS next

Recommended order, without pretending the expert *is* the fly:

1. Keep `ExpertLandingController` as the labelled conventional baseline.
2. Record expert Parquet as the **action** targets.
3. Add retinal encoding of the Three.js (or a dedicated) view into MaleCNS photoreceptors.
4. Run MaleCNS; log descending-neuron activity time-aligned with the expert actions.
5. Train only an **output decoder** (activity → aileron/elevator/rudder/throttle) against those actions.
6. Implement `MaleCNSController` (frozen connectome + decoder) and `TrainedMaleCNSController` as separate classes.
7. Compare success rate and touchdown metrics against this expert, with the HUD stating which computation is in the loop.
