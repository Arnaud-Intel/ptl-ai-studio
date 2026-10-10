@echo off
setlocal
title Panther Lake AI Studio - Setup
cd /d "%~dp0"

REM The setup assistant: a page in the browser that walks through the
REM installation, served from this window by PowerShell (setup\assistant.ps1).
REM It needs nothing installed -- Windows PowerShell and a browser are part of
REM Windows. If the page cannot be served, or you ask for it with
REM "first_launch.bat console", the same steps are done in this window
REM instead (first_launch.ps1).

REM Started from inside the ZIP, Windows extracts this one file to a
REM temporary folder and runs it there, alone: say so, rather than fail on
REM everything that is not beside it.
if not exist "%~dp0uv.lock" goto incomplete
if not exist "%~dp0setup\assistant.ps1" goto incomplete

if /i "%~1"=="console" goto console
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\assistant.ps1" %*
set "rc=%errorlevel%"
if "%rc%"=="9" goto console
goto end

:console
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0first_launch.ps1"
set "rc=%errorlevel%"
goto end

:incomplete
echo.
echo  This file cannot work from here: the rest of the project is not beside it.
echo.
echo  If you opened it from inside the downloaded ZIP: close this window,
echo  right-click the ZIP, choose "Extract All...", then double-click
echo  first_launch.bat in the extracted folder.
echo.
set "rc=1"

:end
echo.
pause
exit /b %rc%
