#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/chatbot"
PID_FILE="$ROOT_DIR/server.pid"

stopped_any=0

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null || true
    fi
    echo "stopped (pid file): $PID"
    stopped_any=1
  fi
fi

# pid file 밖에서 직접 실행된 서버까지 정리합니다.
# conda wrapper / 상대경로 실행 / 직접 python 실행 케이스를 모두 포함합니다.
EXTRA_PIDS="$(
  pgrep -f "server.py" || true
)"
if [[ -n "${EXTRA_PIDS:-}" ]]; then
  while IFS= read -r p; do
    [[ -z "${p:-}" ]] && continue
    kill "$p" 2>/dev/null || true
    sleep 0.2
    if kill -0 "$p" 2>/dev/null; then
      kill -9 "$p" 2>/dev/null || true
    fi
    echo "stopped (scan): $p"
    stopped_any=1
  done <<< "$EXTRA_PIDS"
fi

rm -f "$PID_FILE"
if [[ "$stopped_any" -eq 0 ]]; then
  echo "not running"
fi
