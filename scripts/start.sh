#!/usr/bin/env bash
# Cloud start: JSBSim backend in the background, Vite in the foreground.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  bash "$ROOT/scripts/install.sh"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

python -m fly_pilot.server --host 0.0.0.0 --port 8765 &
BACKEND_PID=$!
cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait until the websocket port is listening without hiding backend crashes.
for _ in $(seq 1 50); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "JSBSim backend exited before it started listening" >&2
    wait "$BACKEND_PID"
    exit 1
  fi
  if python - <<'PY'
import socket
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("127.0.0.1", 8765))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
  then
    break
  fi
  sleep 0.1
done

cd "$ROOT/frontend"
npm run dev -- --host 0.0.0.0 --port 5173
