#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/chatbot"
PID_FILE="$ROOT_DIR/server.pid"
LOG_FILE="$ROOT_DIR/server.log"

echo "[chatbot] status_server.sh"

if [[ ! -f "$PID_FILE" ]]; then
  echo "not running (pid file missing)"
  exit 1
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "${PID:-}" ]]; then
  echo "not running (empty pid file)"
  exit 1
fi

if kill -0 "$PID" 2>/dev/null; then
  echo "running (pid: $PID)"
  echo "log: $LOG_FILE"
  exit 0
fi

echo "not running (stale pid: $PID)"
exit 1
