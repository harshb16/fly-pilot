#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
python -m fly_pilot.brain.prepare "$@"
