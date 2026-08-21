@echo off
title Panther Lake AI Studio
cd /d "%~dp0"
echo Starting Panther Lake AI Studio...
echo Close this window (or press Ctrl+C) to stop the server.
echo.
uv run panther-lake-launcher
pause
