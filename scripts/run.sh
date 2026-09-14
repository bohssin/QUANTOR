#!/usr/bin/env bash
# Start the QUANTOR app. Plan §15.
#
#   ./scripts/run.sh              # http://127.0.0.1:8000
#   ./scripts/run.sh --port 9000
#   ./scripts/run.sh --reload     # develop the UI without restarting
#
# Bound to 127.0.0.1 on purpose. There is no authentication because there is no
# remote: this serves one person's own research on their own machine. Do not
# expose the port.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=8000
HOST=127.0.0.1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --host)   HOST="$2"; shift 2 ;;
    --reload) EXTRA+=(--reload); shift ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No virtualenv found. Run ./scripts/setup.sh first." >&2
  exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "fastapi/uvicorn are missing from .venv. Run ./scripts/setup.sh." >&2
  exit 1
fi

export QUANTOR_LIBRARY="${QUANTOR_LIBRARY:-$ROOT/library}"
mkdir -p "$QUANTOR_LIBRARY"

echo
echo "  QUANTOR"
echo "  library : $QUANTOR_LIBRARY"
echo "  open    : http://$HOST:$PORT"
echo
exec "$ROOT/.venv/bin/uvicorn" app.api.main:app --host "$HOST" --port "$PORT" "${EXTRA[@]}"
