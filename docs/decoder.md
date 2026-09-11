# Trained temporal decoder (Milestone 5)

FLY CONTROL: a **fixed** MaleCNS LIF plus an **external** causal GRU maps
descending-neuron activity onto Cessna inceptors. This is not biological
synaptic learning. `ExpertLandingController` is not in the closed-loop path.

## REAL / MODELED / CONVENTIONAL / LEARNED

| Layer | What |
|---|---|
| **REAL** | JSBSim `c172p`; MaleCNS v1.0 wiring; R1–R6 and DN identities |
| **MODELED** | Python cubemap, R1–R6 spatial layout, LIF dynamics, fly-eye encoder |
| **CONVENTIONAL** | `ExpertLandingController` (training targets only) |
| **LEARNED** | GRU decoder weights. Not connectome plasticity. |

## Control path

```
aircraft pose
  → canonical Python fly-view (cubemap)
    → FlyEyeEncoder
      → actual R1–R6 currents
        → full MaleCNS LIF @ 50 Hz
          → DN 100 ms rates  ⊕  DN 260 ms rates   (≈ 2,628 features)
            → input normalization (training-set mean/std)
              → Linear(F, 256) → LayerNorm → GELU
                → 2-layer GRU (hidden 128, dropout 0.1)
                  → aileron/elevator/rudder tanh, throttle sigmoid
                    → optional first-order slew (α = 0.70, documented)
                      → AircraftControls → JSBSim
```

No aircraft telemetry enters the decoder. No hand-coded runway correction
is applied after the decoder. GRU hidden state persists for the episode and
resets on `reset()`.

## Dataset

`python -m fly_pilot.record_decoder --successes 50 --seed 1000`

Compact parquet columns:

- `episode_id`, `timestep`, `sim_time_s`
- `dn_rates_100_f16`, `dn_rates_260_f16`
- `expert_aileron`, `expert_elevator`, `expert_rudder`, `expert_throttle`

Spawns use `DECODER_SPAWN` (wider lateral / heading / altitude / airspeed /
roll / pitch than Milestone 2, still inside expert recovery). Some episodes
add small bounded **executed-command** pulses; the supervised target remains
the expert's `act()` output. No wind. A few full retinal observing episodes
can be written separately with `--full-replay-output`.

## Training

Episode-level split (~40 / 5 / 5). The first checkpoint trained on
independent 50-step windows (hidden reset each window) collapsed to the
mean action after ~1 s in closed loop. Retraining uses **truncated BPTT
on full episodes** so GRU state persists the way it does in FLY CONTROL.

Causal chunks of 50 neural steps (~1 s) carry hidden state across the
landing. Opening (first 20 s) and flare (last 25 s) are upweighted.
Targets are shifted by 2 neural steps (40 ms) to partly absorb LIF lag.
Huber loss per control plus a small action-smoothness term. AdamW,
modest weight decay, early stopping on validation loss. Best checkpoint
is saved, not the last epoch. Output heads are initialized at the
training-set mean action.

Baselines (offline, not used to fly):

1. constant mean action
2. Ridge on the same DN features

The TBPTT retrain beats Ridge on MAE for three of four inceptors because it
stays close to the mean-action baseline. It does **not** reconstruct the
expert's opening roll/yaw pulses or the flare; Pearson correlations stay
near zero. Closed-loop, that mean-like policy immediately departed the
runway envelope (18/20 out-of-bounds). The shipped `best.pt` is therefore
the first windowed-training checkpoint, which at least produced a brief
opening transient and flew closer to the centerline before crashing. FLY
CONTROL still uses that GRU, never the expert.

## HUD

FLY CONTROL / FIXED MALECNS + TRAINED TEMPORAL DECODER.

Live decoder aileron, elevator, rudder, throttle, MaleCNS spike rate, DN
mean rate, GRU hidden-state norm.

## Checkpoint

`artifacts/decoder/best.pt` plus `.meta.json`. Contains architecture, weights,
DN body-id order, scaler statistics, target conventions, dataset identity,
git commit, seed, metrics. Override path with `FLYPILOT_DECODER_PATH`.
