# Hybrid Fly Guidance

`HYBRID FLY GUIDANCE` is the project's working, explicitly hybrid controller.
It is not the fixed-MaleCNS experiment and it is not evidence that a fly brain
can naturally control an aircraft.

## Why this mode exists

The stricter DN-only experiment froze one global MaleCNS LIF model and trained
an external GRU from all descending-neuron rates directly to four Cessna
inceptors. All four fixed-budget candidates scored `0/20` closed-loop
landings. Diagnostics found very weak DN/action correlation, and DAgger changed
the failure mode without producing a valid landing.

Primary-source review showed that working embodied fly systems use a hierarchy:
a connectome-constrained perception or guidance model produces a small command,
while a separate controller handles the body. FlyVis task-optimizes unknown
dynamics under connectome constraints, and FlyGym maps narrow visual/descending
signals into hybrid lower-level controllers. See
[`research-implementation-review.md`](research-implementation-review.md) for
the pinned papers and source links.

## Implemented path

```text
aircraft telemetry (16 declared fields)
  -> learned observation encoder
  -> five annotated MaleCNS sensory superclasses
  -> five trainable message-passing layers
       edges: signed, incoming-normalized MaleCNS superclass connectivity
  -> descending / efferent / motor superclass readout
  -> temporal GRU
  -> raw bank guidance
  -> ±4° residual inside a conventional lateral safety envelope
  -> conventional glideslope, airspeed, flare, and PID stabilization
  -> JSBSim C172P
```

The graph has 27 nodes, one per retained MaleCNS superclass. Its directed edge
weights are aggregated from all 25,582,938 prepared neuron-to-neuron edges;
presynaptic transmitter signs are retained and each target population is
incoming-normalized. The checkpoint embeds and hashes the resulting topology,
so a fresh checkout can verify and run inference without the 1.1 GB source
dataset or training Parquet.

## Provenance table

| Part | Provenance |
|---|---|
| C172P dynamics and state | **SIMULATED:** JSBSim |
| Population nodes and directed topology | **MEASURED:** MaleCNS v1.0 annotations/connectivity |
| Transmitter sign and population aggregation | **MODELED:** engineering assumptions |
| Observation encoder, message-passing, GRU | **LEARNED:** expert-imitation training |
| Bank residual safety envelope | **CONVENTIONAL:** runway-relative lateral guidance |
| Glideslope, airspeed, flare, rate damping | **CONVENTIONAL:** aircraft control laws |
| Expert at runtime | **ABSENT** |
| Aircraft telemetry at runtime | **PRESENT and disclosed** |

The runtime HUD exposes the raw graph bank command, deployed residual, phase,
guidance targets, and actuator commands. It also states when the residual was
clipped by the safety envelope.

## Reproduce

```bash
# Conventional demonstrations, wide spawn distribution, 20 Hz rows
python -m fly_pilot.record_expert \
  --episodes 100 --seed 5000 --stride 6 --wide-spawns \
  --output data/graph/expert_guidance.parquet

# Train the population-graph guidance model
python -m fly_pilot.train_graph_policy \
  --data data/graph/expert_guidance.parquet \
  --output artifacts/connectome_graph/best.pt

# Closed-loop validation
python -m fly_pilot.evaluate_graph \
  --episodes 20 --seed 6000 \
  --json artifacts/connectome_graph/validation-residual.json

# Portable artifact verification (no local training data required)
python -m fly_pilot.verify_graph_artifact
```

## Ablation history

The implementation retained the intermediate reports because they explain what
made the controller work:

| Variant | Unseen validation landings |
|---|---:|
| Graph directly predicts all four guidance targets | 1/20 |
| Graph bank target + conventional longitudinal guidance | 6/20 |
| Safety-bounded graph bank residual + conventional autoland | 20/20 |

The result therefore demonstrates a reliable **hybrid residual controller**,
not an autonomous connectome-only policy. A necessary follow-up is to replace
telemetry input with calibrated retinotopic visual features and quantify the
residual's causal contribution against zero, shuffled, and disconnected-graph
ablations.
