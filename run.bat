@echo off
title Local RAG & Vector DB Explorer
echo ========================================================
echo   Launching Local RAG & Vector DB Laboratory Sandbox
echo ========================================================
echo.

if not exist venv (
    echo Error: Virtual environment 'venv' not found.
    echo Please build the environment first.
    pause
    exit /b 1
)

echo [1/2] Activating Python virtual environment...
call venv\Scripts\activate.bat

echo [2/2] Launching FastAPI backend server...
echo.
echo --------------------------------------------------------
echo   Server starting at http://127.0.0.1:8000
echo   Opening browser in 2 seconds...
echo --------------------------------------------------------
echo.

# Open browser
start http://127.0.0.1:8000

# Start server
python backend/main.py

pause
