# Fly-brain implementation review for FlyPilot

Research date: 2026-09-16. This review uses primary sources only: papers,
official documentation, and source repositories maintained by the projects'
authors. Its purpose is to identify what is required for a working and
scientifically defensible FlyPilot controller, not to argue that a connectome is
already an executable animal.

## Bottom line

FlyPilot's engineering platform is substantial, but the current controller is
not close to reliable autonomous landing. The `0/20` result is consistent with
an architectural mismatch rather than merely an under-trained GRU:

1. R1-R6 neurons are assigned visual directions by body-ID rank, not by optic
   column. This breaks the retinotopic structure that makes fly motion circuits
   meaningful.
2. The whole CNS is run with one uncalibrated, global LIF parameter set. The
   published models that recover useful visual representations either validate
   a circuit against experiments or task-optimize unknown dynamics while
   keeping anatomical constraints.
3. All descending neurons are treated as a flat feature vector, even though DNs
   are a heterogeneous, functionally organized interface between brain and
   nerve cord.
4. A visual-only representation is asked to imitate four low-level commands
   from an expert that uses hidden aircraft state. Airspeed, angular rates, and
   vertical speed are not generally recoverable without a sufficiently rich and
   calibrated temporal visual representation.
5. Directly mapping fly DNs to Cessna aileron, elevator, rudder, and throttle
   skips the motor hierarchy. Published embodied fly work maps perception to a
   small descending command and lets a separate lower-level controller handle
   stabilization and actuation.

The highest-probability route to a working portfolio project is therefore a new
and explicitly labelled **HYBRID FLY GUIDANCE** mode:

```text
calibrated compound-eye image
  -> task-optimized connectome-constrained visual model
  -> learned runway / optic-flow guidance representation
  -> low-dimensional bank and flight-path targets
  -> conventional aircraft stabilization loops
  -> JSBSim
```

The current fixed-MaleCNS, DN-only, direct-actuator mode should remain available
as a negative research baseline. A hybrid result must not be relabelled as
"the fly brain landed the aircraft." The honest claim is that a
connectome-constrained visual representation supplies high-level guidance to a
conventional aircraft controller.

## What the primary projects actually implement

### FlyWire and MaleCNS: wiring and annotations, not an executable policy

FlyWire's published adult female brain is a wiring diagram: 139,255 neurons,
roughly 50 million chemical synapses, cell annotations, and predicted
neurotransmitters. Its authors explicitly describe simulation as requiring
additional assumptions: physiological strength is commonly approximated from
synapse count, while signs are inferred from transmitter identity. The paper
also notes limitations in synapse detection and sensory-neuron reconstruction.
The connectome is the structural constraint, not a set of neural dynamics or a
motor policy ([FlyWire paper](https://www.nature.com/articles/s41586-024-07558-y)).

MaleCNS improves the situation for embodiment because it contains the brain,
neck connective, and ventral nerve cord in one specimen. Its analysis describes
motor control as distributed and embodied: local sensory-effector loops in the
VNC are coordinated by long-range ascending and descending pathways, rather
than every actuator being commanded independently by the brain
([MaleCNS overview](https://www.janelia.org/node/70079),
[distributed-control paper summary](https://www.janelia.org/publication/distributed-control-circuits-across-a-brain-and-cord-connectome)).

The MaleCNS supplemental repository contains two data products FlyPilot is not
using:

- `optic-column-type-assignments-v1.0.xlsx` identifies L1, R7, and R8 cells by
  optic column and column type. It does not provide calibrated optical axes, but
  it provides the anatomical grouping needed to stop treating body-ID order as
  retinotopy.
- `dnan_cluster_function_*.csv` assigns descending and ascending cells to
  clusters and collects known functions and references. The current release
  includes flight, steering, escape, walking, and other functional labels.

Source: [official MaleCNS supplemental-data README](https://github.com/flyconnectome/2025malecns/blob/67767d2233657983993ff6c2be48e836a935863c/README.md).

**What it does not establish:** a connectome release does not specify membrane
dynamics, sensory encoding, internal state, learning rules, or a mapping from
DNs to Cessna control surfaces.

### Shiu et al.: whole-brain LIF for causal circuit hypotheses

Shiu et al. built a whole-brain FlyWire LIF model and successfully predicted
responses in feeding and antennal-grooming circuits. The implementation uses a
20 ms membrane time constant, 5 ms synaptic-current time constant, 1.8 ms
synaptic delay, 2.2 ms refractory period, and a global 0.275 mV per-synapse
scale. Connectivity and transmitter-derived sign are fixed. Selected sensory
neurons are driven with Poisson input and downstream firing is measured across
30 repeated one-second trials
([paper](https://www.nature.com/articles/s41586-024-07763-9),
[reference implementation](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py)).

This is strong evidence that a simple connectome-constrained LIF model can be
useful for particular, well-defined sensorimotor pathways. It is not evidence
that an arbitrary continuous video stream through a global LIF model yields a
general-purpose control state. The paper is explicit that the model assumes
zero baseline firing, treats every neuron identically, and omits morphology,
gap junctions, non-spiking neurons, internal state, receptor-specific effects,
neuromodulation, and extrasynaptic signalling. The demonstrated use is
perturbation and hypothesis generation, not autonomous closed-loop behaviour.

**Lesson for FlyPilot:** if the Shiu model is cited as the physiological basis,
its synaptic state, delay, refractory period, and numerical integration capable
of resolving millisecond-scale dynamics cannot be replaced by a 50 Hz
instantaneous update without calling
that replacement a separate engineered model. More importantly, Shiu et al.
validate specific pathways with known inputs and outputs; FlyPilot should do the
same before asking for closed-loop control.

### FlyVis: fixed anatomy plus trained unknown dynamics

FlyVis is the official implementation of Lappalainen et al.'s
connectome-constrained deep mechanistic network. It models 64 visual cell types,
45,669 neurons, and about 1.5 million connections on a retinotopic hexagonal
lattice. Each node corresponds to an identified neuron and edges exist only
where the connectome allows them. The model uses passive point neurons with
instantaneous graded release, not a whole-brain spiking emulation
([paper](https://www.nature.com/articles/s41586-024-07939-3),
[dynamics source](https://github.com/TuragaLab/flyvis/blob/92b3845cc426dd309a1a0e1b3890156c42e14021/flyvis/network/dynamics.py)).

The important implementation choice is what FlyVis learns. Synapse counts and
signs remain fixed, while resting potentials and time constants are trainable by
cell type and a non-negative unitary synaptic strength is trainable per
source-type/target-type pair
([node configuration](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021/flyvis/config/network/node_config),
[edge configuration](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021/flyvis/config/network/edge_config)).
The connectome-constrained network and a feed-forward decoder are trained
end-to-end on video sequences to estimate optic flow. The decoder sees
instantaneous downstream activity, so temporal motion information has to be
computed by the recurrent visual network itself. The paper's ablations show
that the connectome alone, without task optimization, does not recover direction
selectivity reliably.

**What it does not claim:** FlyVis models a visual motion pathway, not an entire
brain, and its trained parameters are hypotheses about unknown physiology. It
does not output motor commands.

**Lesson for FlyPilot:** freezing arbitrary neural dynamics and training only a
large external GRU discards the strategy that makes FlyVis work. The closest
scientifically grounded alternative is to keep graph topology, signs, and
relative synapse counts constrained while training a small number of
cell-type-shared dynamics parameters on an explicitly visual objective.

### NeuroMechFly / FlyGym: hierarchical control, calibrated retina, and narrow actions

FlyGym is an embodied fly platform with MuJoCo physics, compound-eye vision,
olfaction, mechanosensory feedback, and explicit brain/VNC hierarchy
([official documentation](https://neuromechfly.org/),
[NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y)).
Its retina applies a fisheye transform, bins pixels onto a hexagonal ommatidial
lattice, and averages intensity within each ommatidium. The two eyes cover about
270 degrees horizontally with about 17 degrees of overlap; pale and yellow
ommatidia receive different blue/green sensitivities. This is a structured eye
model, not one arbitrary visual direction per photoreceptor cell.

The published multimodal navigation example is deliberately hierarchical. A
small graph-convolutional visual model is first supervised to predict object
presence, angle, distance, azimuth, and relative size from the ommatidial graph
([source](https://github.com/NeLy-EPFL/nmf2-paper/blob/1597277deff6b97faf6e021f05afd12ff39fa23e/integrated_task/vision_model.py)).
A Soft Actor-Critic policy then receives those visual features, bilateral odor,
and the previous action and emits one turning bias. A hybrid locomotor controller
maps that bias to left/right descending drive and handles CPGs, adhesion, and
joint control. Training used 500,000 environment steps and dense task rewards
([training source](https://github.com/NeLy-EPFL/nmf2-paper/blob/1597277deff6b97faf6e021f05afd12ff39fa23e/integrated_task/preprint_trial/train_navigation_task.py),
[environment source](https://github.com/NeLy-EPFL/nmf2-paper/blob/1597277deff6b97faf6e021f05afd12ff39fa23e/integrated_task/preprint_trial/rl_navigation.py)).

Its connectome-constrained fly-following demonstration is equally revealing. It
runs a pretrained FlyVis network, z-scores activity from selected transmedullary
and T4/T5-related cell types against baseline, engineers an object mask and
center deviation from those responses, and maps that deviation to two
left/right descending drives. A learned head stabilizer and conventional hybrid
walking controller perform the lower-level work
([closed-loop source](https://github.com/NeLy-EPFL/flygym/blob/c7affce924cb1c6add16619adf83be5c6b223e89/flygym/examples/vision/follow_fly_closed_loop.py),
[implementation notes](https://github.com/NeLy-EPFL/flygym/blob/c7affce924cb1c6add16619adf83be5c6b223e89/flygym/examples/vision/README.md)).

**What it does not claim:** the connectome-constrained model does not directly
control individual leg joints, and the navigation policy is not a whole-brain
emulation. Perception, descending guidance, head stabilization, and locomotion
are separate modules with different learned or engineered provenance.

**Lesson for FlyPilot:** emulate this separation. A connectome-informed visual
module should issue a small guidance command; a lower-level aircraft controller
should stabilize the Cessna.

### FlyBrainLab and Neurokernel: executable models remain explicit models

FlyBrainLab integrates anatomical, genetic, physiological, and computational
models and lets researchers compare executable circuit hypotheses. It exists
specifically to go beyond a connectome by attaching explicit neuron, synapse,
and sensory-transduction models to circuits and validating alternatives
([FlyBrainLab paper](https://elifesciences.org/articles/62362),
[official repository](https://github.com/FlyBrainLab/FlyBrainLab)).
Neurokernel divides the nervous system into local processing units with defined
interfaces so independently developed models can be composed. Its retina/lamina
example uses a stochastic phototransduction model and verifies that visual
signals and pathway inversion survive across modules
([Neurokernel paper](https://doi.org/10.1371/journal.pone.0146581),
[integration RFC](https://neurokernel.github.io/rfc/nk-rfc4.pdf)).

**What it does not claim:** turning a reconstructed graph into generic LIF units
does not by itself recover its function. The executable circuit is a documented,
parameterized hypothesis that must be tested.

### Fly64: an engineered causal demo, not recovered fly behaviour

Fly64 is the closest public project to FlyPilot's current architecture. It runs
MaleCNS at 50 Hz, injects engineered cubemap-derived current into
photoreceptors, and uses hand-written readouts from named DNs: DNg100 for
forward motion, bilateral DNa02/DNg13 differences for steering, and DNp01/DNp10
bursts for jumping. It explicitly labels the global dynamics, retinal encoding,
and motor mapping as engineering choices, uses causal frozen/disconnected-frame
tests, and makes no natural-fly-behaviour claim
([technical notes](https://github.com/ornata/fly/blob/f2f4114e53eaa326e54129f27a5383f93c6957af/docs/technical-notes.md)).

The key point is that Fly64 obtains a visible loop by using functional prior
knowledge and a low-dimensional hand mapping. It does not show that all-DN
activity contains enough information for end-to-end imitation of an unrelated
plant's expert controller. It is a useful demo architecture, not validation of
FlyPilot's current learning problem.

## Direct comparison with FlyPilot

| Design choice | Established implementations | Current FlyPilot consequence |
|---|---|---|
| Retinotopy | Hexagonal ommatidia or anatomical optic columns; repeated receptor cells share a visual unit | Body-ID sorting assigns unrelated directions to connected cells, likely destroying coherent motion signals |
| Neural dynamics | Experiment-calibrated for a narrow circuit, or task-optimized under connectome constraints | One global, engineered 50 Hz LIF configuration is frozen before its representation is validated |
| Readout | Named functional populations, cell-type maps, object/flow features, or a small descending command | 1,314 heterogeneous DNs × two windows are presented as an unstructured 2,628-D vector |
| Motor hierarchy | Brain-level steering plus VNC/reflex/CPG control | DN features directly predict four raw Cessna inceptors |
| Training task | Optic flow, object geometry, or closed-loop task reward | Behaviour cloning targets a telemetry-based PID action that may not be a function of the visual/DN input |
| Validation | Neural tuning, known pathway response, ablations, task success | Retinal hashes and aggregate DN changes establish causality, but not whether the needed state is represented |

DAgger can correct covariate shift, but it cannot fix an information problem. If
different aircraft states produce effectively indistinguishable DN vectors yet
require different expert commands, adding more expert labels at those states
drives the model toward an average action. The current negative R-squared values
and mean-baseline behaviour are exactly what this failure mode predicts.

## Recommended implementation, in order

### 1. Preserve the negative baseline and stop tuning the same input

Keep the existing checkpoint, data split, offline report, and `0/20`
closed-loop evaluation under a clear name such as **DN-only research
baseline**. Finish any already-running fixed experiment once, but do not begin
an open-ended GRU/DAgger search on the same representation.

Acceptance criterion: the old result is reproducible and cannot be confused
with the new hybrid mode.

### 2. Repair the eye mapping before changing the decoder

Extend MaleCNS preparation to hash and ingest the official optic-column
workbook. Build ommatidial groups from L1/R7/R8 column labels and derive R1-R6
membership from their anatomical connections to the column, with confidence
and fallback recorded per cell. Cells belonging to one ommatidium should sample
the same optical direction; receptor subtype should determine channel/spectral
processing, not a new direction. Use a hexagonal eye lattice and an explicit,
versioned mapping from column coordinates to spherical rays. The optical-axis
mapping remains modelled and must be labelled as such.

Add representation tests, not just hash tests:

- horizontal yaw produces signed left/right motion responses;
- roll produces the expected vertical/horizontal optic-flow rotation;
- forward translation produces symmetric expansion;
- freezing the scene suppresses temporal motion responses;
- shuffling column assignments destroys direction decoding.

Acceptance criterion: a small linear decoder can recover the sign and magnitude
of controlled yaw, roll, and forward optic flow on held-out trajectories.

### 3. Use a task-optimized visual circuit

The practical first implementation should integrate the official pretrained
FlyVis network as an optional visual backend and adapt the existing rendered
eye frames to its hexagonal input ordering. This supplies a tested
connectome-constrained motion representation while MaleCNS remains available
for comparison.

In parallel, the research extension can extract the MaleCNS subgraph from
photoreceptors through optic-lobe/visual-projection cells to functionally
relevant DNs. Keep topology, signs, and relative synapse counts fixed, but learn
cell-type-shared time constants, biases, and source-type/target-type gains on an
optic-flow or runway-geometry objective, following FlyVis. Call this a
**task-optimized connectome-constrained model**, not fixed MaleCNS physiology.

Acceptance criterion: held-out visual targets beat constant and pixels-only
linear baselines, and intermediate direction-selective cell types pass tuning
tests.

### 4. Train perception targets before actions

Use simulator truth only as training labels for a compact visual-state head:

- runway-center bearing or image-space offset;
- runway heading error;
- horizon roll and pitch;
- optical expansion / time-to-contact;
- runway confidence or visibility;
- optionally a recurrent estimate of glideslope trend.

No JSBSim value enters the model at inference. This makes observability
measurable: each target must show useful held-out correlation/R-squared before
it is trusted by a controller.

Acceptance criterion: every control-relevant target has a stated held-out
metric and uncertainty/fallback behaviour.

### 5. Add `HYBRID FLY GUIDANCE`, not another direct-actuator decoder

Map the learned visual state to two or three slow guidance quantities, for
example desired bank, desired flight-path angle, and runway-confidence/flare
state. Reuse conventional inner loops for attitude-rate damping, airspeed and
throttle, coordination, and flare execution. This mirrors FlyGym's descending
drive to hybrid motor-controller boundary.

The WebSocket/HUD/README provenance should say:

- **CONNECTOME-CONSTRAINED / LEARNED:** visual representation and guidance;
- **CONVENTIONAL:** aircraft stabilization, airspeed control, and safety limits;
- **NOT USED AT INFERENCE:** runway-relative simulator truth and expert actions.

Acceptance criterion: the hybrid mode lands reliably and an ablation replacing
the visual guidance with zero or shuffled features significantly degrades
performance.

### 6. Use functional DN priors for the MaleCNS experiment

Do not pool every DN indiscriminately. Import the official DN/AN functional
table and report separate type-level bilateral pools for flight, steering,
rotational-flow, escape, and unrelated behaviours. Candidate pools include the
published `flight` and `steering` labels and `DNp20`, which is annotated as
tuned to rotational flow fields. The selection is a biological prior, not a
learned discovery, and must be recorded in artifact metadata.

Compare four readouts on identical data:

1. all individual DNs;
2. type-pooled DNs;
3. flight/steering functional pools;
4. visual-cell outputs before the DN bottleneck.

Acceptance criterion: select a readout only if it predicts the visual-state
targets out of sample. If DNs lose information available upstream, retain that
as a result rather than hiding it.

### 7. Fine-tune high-level guidance in closed loop

After supervised perception works, use either high-level DAgger or a small
closed-loop RL policy. The action space should be guidance targets, not four
surfaces. Use randomized spawn, lighting, texture, wind, and camera perturbation
with a curriculum from straight stabilized approaches to lateral/heading
offsets and finally flare. Rank models by held-out closed-loop success; keep the
final 100 seeds sealed.

Useful reward components are progress toward threshold, centerline and heading
alignment, glideslope error, excessive bank/sink penalties, smooth guidance,
and a terminal touchdown score. These are training signals for an external
policy, not evidence of fly learning.

### 8. Require causal and provenance ablations

For every claimed connectome contribution, run:

- live vision versus frozen vision;
- intact versus shuffled retinotopy;
- intact versus disconnected visual graph;
- trained versus random dynamics;
- real guidance features versus shuffled features;
- pixels-only and conventional-expert baselines.

The final report should distinguish task success from biological validity. A
landing rate demonstrates controller performance; tuning and perturbation
tests establish what the connectome-constrained representation contributed.

## Recommended portfolio claim

If the hybrid mode succeeds:

> Built a real-time JSBSim research platform that couples a calibrated
> compound-eye renderer to a connectome-constrained fly visual model. A learned
> visual-guidance policy supplies low-dimensional approach commands to a
> conventional aircraft stabilizer; causal ablations quantify the contribution
> of the neural representation. A stricter fixed-MaleCNS DN-only controller was
> evaluated separately and failed 0/20 approaches.

If only the platform and negative baseline remain, present it as a negative
result, not a hidden failure. Both versions are stronger and more credible than
claiming that a generic whole-connectome LIF model directly learned to fly a
Cessna.

## Decisions that should not be reversed

- Do not feed telemetry into a model still labelled DN-only.
- Do not blend the expert into autonomous control.
- Do not replace the connectome with an ordinary neural network while retaining
  the fly-control label.
- Do not call trained synaptic/dynamics parameters measured physiology.
- Do not describe FlyVis, FlyGym, Fly64, FlyBrainLab, or Shiu et al. as full fly
  cognition or general-purpose autonomous agents; their primary sources make
  much narrower claims.
