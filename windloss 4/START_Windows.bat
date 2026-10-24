@echo off
REM Double-click this file on Windows to start the app.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install it from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
if not exist ".venv" (
  echo First start: setting up - takes 1-3 minutes, only once...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip -q
  .venv\Scripts\python -m pip install -r requirements.txt -q
)
echo Starting the app - your browser will open. Keep this window open while you use it.
.venv\Scripts\python -m streamlit run app.py
pause
