@echo off
setlocal
title Panther Lake AI Studio - First launch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0first_launch.ps1"
set "rc=%errorlevel%"
echo.
pause
exit /b %rc%
