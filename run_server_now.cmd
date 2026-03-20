@echo off
cd /d "%~dp0"
echo 서버 시작 중...
wsl -e bash -lc "cd /home/doohyeon/chatbot && bash start_server.sh && bash status_server.sh"
echo.
echo 접속 주소: http://localhost:8000
echo 브라우저에서 위 주소로 접속하세요.
start http://localhost:8000
pause
