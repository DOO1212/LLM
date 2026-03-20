@echo off
chcp 65001 >nul
echo 서버 시작 중... (WSL 출력이 아래에 나와야 합니다)
echo.
wsl.exe -e bash -lc "bash ~/chatbot/start_server.sh && bash ~/chatbot/status_server.sh"
set WSL_EXIT=%ERRORLEVEL%
echo.
if %WSL_EXIT% neq 0 (
  echo [오류] WSL 종료 코드: %WSL_EXIT%
  echo   - Ubuntu 앱을 열고:  bash ~/chatbot/start_server.sh
  echo   - 또는 conda 환경 llm, chatbot 폴더에서 python server.py
) else (
  echo WSL 스크립트는 정상 종료했습니다.
)
echo.
echo 접속 주소: http://localhost:8000
echo 브라우저에서 위 주소로 접속하세요.
echo.
pause
