@echo off
title Panther Lake AI Studio
cd /d "%~dp0"

REM The installer (uv), wherever the setup left it: on the PATH, where winget
REM or uv's own installer puts it, or in this folder (.tools\uv) for a laptop
REM that had none. A PATH changed by an installation is not seen by a window
REM that was already open, hence the list.
set "PTL_UV="
for /f "delims=" %%i in ('where uv 2^>nul') do if not defined PTL_UV set "PTL_UV=%%i"
if not defined PTL_UV if exist "%~dp0.tools\uv\uv.exe" set "PTL_UV=%~dp0.tools\uv\uv.exe"
if not defined PTL_UV if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" set "PTL_UV=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
if not defined PTL_UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "PTL_UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined PTL_UV goto notinstalled

echo Starting Panther Lake AI Studio...
echo Close this window (or press Ctrl+C) to stop the server.
echo.
REM Offline first. uv re-syncs the environment on every run, and on a
REM machine with no connection that attempt is what stops the app from
REM starting at all. --offline uses what is installed and uv's cache; if
REM something genuinely has to be fetched, the second attempt does that.
"%PTL_UV%" run --offline panther-lake-launcher %*
set "rc=%errorlevel%"
if "%rc%"=="3" goto upgrading
if "%rc%"=="0" goto done

echo.
echo Could not start from what is installed -- trying again with the network...
"%PTL_UV%" run panther-lake-launcher %*
set "rc=%errorlevel%"
if "%rc%"=="3" goto upgrading
if not "%rc%"=="0" (
  echo.
  echo Startup failed -- see the message above.
  echo If a copy is already running, stop_launcher.bat will stop it.
  echo To repair the installation, run first_launch.bat.
)
goto done

:notinstalled
echo.
echo  The Studio is not installed on this laptop yet.
echo  Double-click first_launch.bat in this folder: it installs everything, step by step.
echo.
goto done

:upgrading
echo.
echo Upgrading: the new version starts in a new window once it is installed.
echo This window can be closed.
timeout /t 10 >nul
exit /b 0

:done
pause
