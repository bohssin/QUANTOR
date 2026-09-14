#!/usr/bin/env bash
# Start/stop/restart the dev server via a pidfile.
#
#   ./scripts/serverctl.sh start|stop|restart|status
#
# A pidfile rather than `pkill -f uvicorn`, because that pattern also matches
# the shell that ran it — including a CI or agent shell whose own command line
# contains the string — and kills the caller instead of the server.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${QUANTOR_PIDFILE:-$ROOT/.server.pid}"
LOG="${QUANTOR_LOG:-$ROOT/.server.log}"
PORT="${QUANTOR_PORT:-8000}"
HOST="${QUANTOR_HOST:-127.0.0.1}"
export QUANTOR_LIBRARY="${QUANTOR_LIBRARY:-$ROOT/library}"

running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

start() {
  if running; then echo "already running (pid $(cat "$PIDFILE"))"; return 0; fi
  mkdir -p "$QUANTOR_LIBRARY"
  setsid "$ROOT/.venv/bin/uvicorn" app.api.main:app \
      --host "$HOST" --port "$PORT" --app-dir "$ROOT" \
      > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do
    if curl -sf -o /dev/null "http://$HOST:$PORT/api/health"; then
      echo "started (pid $(cat "$PIDFILE")) on http://$HOST:$PORT"
      return 0
    fi
    sleep 0.5
  done
  echo "failed to become healthy; last log lines:" >&2
  tail -20 "$LOG" >&2
  return 1
}

stop() {
  if ! running; then echo "not running"; rm -f "$PIDFILE"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -9 "$pid" 2>/dev/null
  rm -f "$PIDFILE"
  echo "stopped"
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)
    if running; then echo "running (pid $(cat "$PIDFILE"))"; else echo "stopped"; fi ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
