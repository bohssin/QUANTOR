#!/usr/bin/env bash
# Start the QUANTOR app. Plan §15.
#
#   ./scripts/run.sh              # http://quantor:2026
#   ./scripts/run.sh --port 9000
#   ./scripts/run.sh --host 127.0.0.1   # loopback only
#   ./scripts/run.sh --reload     # develop the UI without restarting
#
# Binds 0.0.0.0:2026 so the machine answers to its own name — open
# http://quantor:2026 if the host is called `quantor`, or add a hosts entry:
#
#   Windows  C:\Windows\System32\drivers\etc\hosts   (edit as Administrator)
#   Linux    /etc/hosts
#
#       127.0.0.1   quantor
#
# There is no authentication, because this was built for one person's own
# research. 0.0.0.0 means anyone who can reach the machine can drive it, so on
# an untrusted network use `--host 127.0.0.1` instead.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=2026
HOST=0.0.0.0
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
if [ "$HOST" = "0.0.0.0" ]; then
  echo "  open    : http://quantor:$PORT   (or http://localhost:$PORT)"
  echo "  note    : reachable from your network — use --host 127.0.0.1 to keep it local"
else
  echo "  open    : http://$HOST:$PORT"
fi
echo
exec "$ROOT/.venv/bin/uvicorn" app.api.main:app --host "$HOST" --port "$PORT" "${EXTRA[@]}"
