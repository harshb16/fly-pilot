#!/usr/bin/env bash
# Idempotent Cloud / local bootstrap. Do not start long-running servers here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! python3 -c "import venv" >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev
  else
    echo "python3 venv module is missing; install python3-venv" >&2
    exit 1
  fi
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  python3 -m venv "$ROOT/.venv"
fi

"$ROOT/.venv/bin/python" -m pip install -U pip
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT/backend[dev]"

cd "$ROOT/frontend"
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

echo "Install complete."
