# MaleCNS local cache

This directory holds the official MaleCNS v1.0 tables and the compact
representation built by:

```bash
python -m fly_pilot.brain.prepare
```

Nothing in `raw/` or `prepared/` is committed. Re-running prepare is
idempotent: matching SHA-256 files are reused.

See `docs/malecns.md` for provenance, hashes, and scientific caveats.
