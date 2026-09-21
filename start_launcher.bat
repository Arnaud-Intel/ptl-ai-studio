@echo off
title Panther Lake AI Studio
cd /d "%~dp0"
echo Starting Panther Lake AI Studio...
echo Close this window (or press Ctrl+C) to stop the server.
echo.
REM Offline first. uv re-syncs the environment on every run, and on a
REM machine with no connection that attempt is what stops the app from
REM starting at all. --offline uses what is installed and uv's cache; if
REM something genuinely has to be fetched, the second attempt does that.
uv run --offline panther-lake-launcher %*
set "rc=%errorlevel%"
if "%rc%"=="3" goto upgrading
if "%rc%"=="0" goto done

echo.
echo Could not start from what is installed -- trying again with the network...
uv run panther-lake-launcher %*
set "rc=%errorlevel%"
if "%rc%"=="3" goto upgrading
if not "%rc%"=="0" (
  echo.
  echo Startup failed -- see the message above.
  echo If a copy is already running, stop_launcher.bat will stop it.
)
goto done

:upgrading
echo.
echo Upgrading: the new version starts in a new window once it is installed.
echo This window can be closed.
timeout /t 10 >nul
exit /b 0

:done
pause
