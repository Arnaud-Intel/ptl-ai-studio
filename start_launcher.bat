@echo off
title Panther Lake AI Studio
cd /d "%~dp0"
echo Starting Panther Lake AI Studio...
echo Close this window (or press Ctrl+C) to stop the server.
echo.
uv run panther-lake-launcher
if errorlevel 1 (
  echo.
  echo Startup failed -- see the message above.
  echo If a copy is already running, stop_launcher.bat will stop it.
)
pause
