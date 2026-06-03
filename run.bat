@echo off
REM AI Squat Coach — Windows launcher
cd /d "%~dp0"
echo ============================================
echo   AI Squat Coach — Windows
echo ============================================
echo.
python main.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Something went wrong. Make sure:
    echo   - Python 3.10+ is installed
    echo   - You ran: pip install -r requirements.txt
    echo   - Your webcam is not in use by another app
    pause
)
