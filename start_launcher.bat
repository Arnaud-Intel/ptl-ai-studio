@echo off
title Panther Lake AI Studio
cd /d "%~dp0"
echo Starting Panther Lake AI Studio...
echo Close this window (or press Ctrl+C) to stop the server.
echo.
uv run panther-lake-launcher %*
if %errorlevel%==3 (
  echo.
  echo Upgrading: the new version starts in a new window once it is installed.
  echo This window can be closed.
  timeout /t 10 >nul
  exit /b 0
)
if errorlevel 1 (
  echo.
  echo Startup failed -- see the message above.
  echo If a copy is already running, stop_launcher.bat will stop it.
)
pause
