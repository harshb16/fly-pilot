# Decoder-training recordings

Compact DN-rate → expert-action pairs. No retinal blobs. No aircraft
telemetry as decoder input.

```bash
python -m fly_pilot.record_decoder --successes 50 --seed 1000 --output data/decoder/expert_dn_controls.parquet
python -m fly_pilot.train_decoder --data data/decoder/expert_dn_controls.parquet --output artifacts/decoder/best.pt
python -m fly_pilot.evaluate_decoder --checkpoint artifacts/decoder/best.pt --data data/decoder/expert_dn_controls.parquet
python -m fly_pilot.evaluate_fly --episodes 20 --seed 2000
```

Targets are `ExpertLandingController` inceptors (classical autopilot).
The trained GRU is an external decoder, not biological synaptic learning.
