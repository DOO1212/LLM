#!/usr/bin/env bash
# 서버 시작 후 결과를 run_report.txt에 저장 (접속 확인용)
set -e
cd /home/doohyeon/chatbot
REPORT="$PWD/run_report.txt"
echo "=== $(date) ===" > "$REPORT"
bash start_server.sh >> "$REPORT" 2>&1
sleep 2
bash status_server.sh >> "$REPORT" 2>&1
ss -tlnp 2>/dev/null | grep -E '8000|LISTEN' >> "$REPORT" 2>&1 || true
echo "접속 주소: http://localhost:8000" >> "$REPORT"
echo "done" >> "$REPORT"
