#!/usr/bin/env bash
# Set up QUANTOR to run locally. Plan §14.
#
#   ./scripts/setup.sh
#
# Creates a venv with the engine + MCP dependencies, writes a .mcp.json that
# points at THAT interpreter with absolute paths, and runs the doctor.
#
# The interpreter matters more than it looks: when .mcp.json names a python
# without `mcp` installed, the server exits instantly and Claude Code reports
# only CONNECTION_CLOSED — no traceback, nothing naming the cause. Using the
# venv's python everywhere is what avoids that.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="${QUANTOR_VENV:-$ROOT/.venv}"

pick_python() {
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  echo "ERROR: need Python 3.11 or newer on PATH." >&2
  exit 1
}

PY="$(pick_python)"
echo "==> Python: $PY ($($PY --version 2>&1))"

if [ ! -d "$VENV" ]; then
  echo "==> Creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi

echo "==> Installing dependencies"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet mcp numpy numba pyarrow pandas pytest
# The app (browser UI + agent sidebar). httpx is what the API tests drive.
"$VENV/bin/pip" install --quiet fastapi "uvicorn[standard]" websockets httpx

echo "==> Writing .mcp.json"
"$VENV/bin/python" quantor_mcp/doctor.py --write-config

echo "==> Running tests"
"$VENV/bin/python" -m pytest tests/ -q

echo
"$VENV/bin/python" quantor_mcp/doctor.py
STATUS=$?

cat <<EOF

Next:
  1. Open the app:
       ./scripts/run.sh              # then http://127.0.0.1:8000

     Load your data on the Data page. A tick CSV needs a timeframe and, if its
     clock is not UTC, the GMT offset — a GMT+3 export is 3. Files above
     ~256 MB stream in bounded memory; bars cache at M1 so every coarser
     timeframe afterwards is free.

  2. Sign in to LuxAlgo (free account; the token is per-machine):
       npx -y @luxalgo/mcp login

  3. Or drive it from a terminal session instead of the browser:
       claude --mcp-config .mcp.json \\
         --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"

       "Load ticks.csv as 'xau' at M15, write an EMA-crossover strategy
        with an ATR stop, backtest it, run control_test on it, then
        validate_run it and tell me honestly whether it generalized."

Re-check anytime with:
  $VENV/bin/python quantor_mcp/doctor.py
EOF
exit $STATUS
