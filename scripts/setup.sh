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

echo "==> Writing .mcp.json"
"$VENV/bin/python" quantor_mcp/doctor.py --write-config

echo "==> Running tests"
"$VENV/bin/python" -m pytest tests/ -q

echo
"$VENV/bin/python" quantor_mcp/doctor.py
STATUS=$?

cat <<EOF

Next:
  1. Sign in to LuxAlgo (free account; the token is per-machine):
       npx -y @luxalgo/mcp login

  2. Start a session:
       claude --mcp-config .mcp.json \\
         --allowedTools "mcp__quantor-engine__*,mcp__luxalgo__*,Read,Edit"

  3. Ask it to load your data and write a strategy:
       "Load ticks.csv as 'xau' at M15, write an EMA-crossover strategy
        with an ATR stop, backtest it, then validate_run it."

Re-check anytime with:
  $VENV/bin/python quantor_mcp/doctor.py
EOF
exit $STATUS
