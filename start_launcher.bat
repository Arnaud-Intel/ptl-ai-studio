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
REM starting at all. --offline uses what is installed and uv's cache; only
REM if something genuinely has to be fetched is the network used.
REM That is asked with a command that does nothing, before the Studio is
REM started: a Studio that ends -- stopped by stop_launcher.bat, say -- must
REM not be taken for one that could not start, and be started again.
"%PTL_UV%" run --offline python -c "pass" >nul 2>&1
if errorlevel 1 goto online
"%PTL_UV%" run --offline panther-lake-launcher %*
set "rc=%errorlevel%"
goto ended

:online
echo Something is missing from what is installed -- fetching it...
echo.
"%PTL_UV%" run panther-lake-launcher %*
set "rc=%errorlevel%"

:ended
if "%rc%"=="3" goto upgrading
if "%rc%"=="0" goto done
echo.
echo The Studio has stopped.
echo If it did not start at all, the message above says why: when a copy is
echo already running, stop_launcher.bat stops it; to repair the installation,
echo run first_launch.bat.
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
