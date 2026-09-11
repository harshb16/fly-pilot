#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
exec python -m fly_pilot.server --host 0.0.0.0 --port 8765
