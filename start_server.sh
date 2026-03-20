#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/chatbot"
PID_FILE="$ROOT_DIR/server.pid"
LOG_FILE="$ROOT_DIR/server.log"
START_ERR_FILE="$ROOT_DIR/server_start_error.log"

# 터미널/Windows→WSL에서도 무조건 한 줄은 보이게
echo "[chatbot] start_server.sh (HOME=$HOME, ROOT=$ROOT_DIR)"

# pid 파일이 없거나 오래돼도 실제 프로세스가 떠 있으면 중복 실행을 막습니다.
RUNNING_PID="$(pgrep -f "server.py" | head -n 1 || true)"
if [[ -n "${RUNNING_PID:-}" ]]; then
  echo "$RUNNING_PID" > "$PID_FILE"
  echo "already running (pid: $RUNNING_PID)"
  echo "log: $LOG_FILE"
  exit 0
fi

cd "$ROOT_DIR"

CONDA_BIN=""
for c in \
  "$HOME/anaconda3/bin/conda" \
  "$HOME/miniconda3/bin/conda" \
  "$HOME/miniforge3/bin/conda" \
  "/opt/conda/bin/conda"
do
  if [[ -x "$c" ]]; then
    CONDA_BIN="$c"
    break
  fi
done

if [[ -z "$CONDA_BIN" ]]; then
  echo "conda binary not found (anaconda3/miniconda3/miniforge3/opt)" | tee "$START_ERR_FILE"
  exit 1
fi

# 주의: $! 는 conda 래퍼 PID일 수 있어 곧 죽고, status_server가 실패할 수 있음.
# 실제 python server.py PID를 잠시 기다렸다가 기록합니다.
nohup "$CONDA_BIN" run -n llm python server.py > "$LOG_FILE" 2>&1 < /dev/null &
sleep 1
NEW_PID=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  NEW_PID="$(pgrep -f "server.py" | head -n 1 || true)"
  if [[ -n "${NEW_PID:-}" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "${NEW_PID:-}" ]]; then
  echo "ERROR: server.py 프로세스가 뜨지 않았습니다. 로그 확인:"
  echo "--- tail $LOG_FILE ---"
  tail -n 30 "$LOG_FILE" 2>/dev/null || echo "(로그 파일 없음)"
  exit 1
fi

echo "$NEW_PID" > "$PID_FILE"
echo "started (pid: $NEW_PID)"
echo "log: $LOG_FILE"
echo "접속 주소: http://localhost:8000  (첫 기동 시 1~2분 후 접속 가능)"
