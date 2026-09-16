# Decoder-training recordings

Compact DN-rate → expert-action pairs. No retinal blobs. No aircraft
telemetry as decoder input.

```bash
python -m fly_pilot.record_decoder --successes 50 --seed 1000 --output data/decoder/expert_dn_controls.parquet
python -m fly_pilot.record_dagger --checkpoint artifacts/decoder/baseline.pt --episodes 25 --seed 1100 --output data/decoder/dagger-round-1.parquet
python -m fly_pilot.record_dagger --checkpoint artifacts/decoder/baseline.pt --episodes 25 --seed 1200 --output data/decoder/dagger-round-2.parquet
python -m fly_pilot.train_decoder \
  --data data/decoder/expert_dn_controls.parquet \
  --data data/decoder/dagger-round-1.parquet \
  --data data/decoder/dagger-round-2.parquet \
  --output artifacts/decoder/best.pt
python -m fly_pilot.evaluate_decoder \
  --checkpoint artifacts/decoder/best.pt \
  --data data/decoder/expert_dn_controls.parquet \
  --data data/decoder/dagger-round-1.parquet \
  --data data/decoder/dagger-round-2.parquet
python -m fly_pilot.evaluate_fly --episodes 20 --seed 3000 --json artifacts/decoder/validation-eval.json
python -m fly_pilot.evaluate_fly --episodes 100 --seed 4000 --json artifacts/decoder/final-eval.json
```

Targets are `ExpertLandingController` inceptors (classical autopilot).
The trained GRU is an external decoder, not biological synaptic learning.
During DAgger collection, FLY CONTROL remains the only authority: the expert
labels the visited state but its command is never applied or blended. Parquet
files and their sidecars remain gitignored; only the selected checkpoint,
split, metadata, and evaluation reports are committed.
