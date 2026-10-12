@echo off
goto main
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
exit /b 0

REM The lines of colons above are where the window of an older version comes
REM back to. cmd reads a command file by position, a line at a time: a window
REM that started the Studio before an upgrade is still open after it, and
REM carries on in THIS file at the place it had reached in ITS own -- which
REM was the last line of each of them, "exit /b 0", at byte 355, 1041, 2101 or
REM 2397. A line of colons is a label and does nothing, wherever in it the
REM reading starts, so such a window reads down to the "exit /b 0" above and
REM closes. Nothing may be put before those lines, nor taken from them.

:main
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
REM
REM Where the OpenVINO engine is installed, uv is told to look after it too
REM (--extra openvino). Without that it re-syncs everything but OpenVINO,
REM and a copy updated by hand -- a git pull, a ZIP unpacked over it -- kept
REM the OpenVINO of the version before, with nothing saying so.
set "PTL_EXTRA="
if exist "%~dp0.venv\Lib\site-packages\openvino\__init__.py" set "PTL_EXTRA=--extra openvino"
set "PTL_OFFLINE=--offline"
"%PTL_UV%" run --offline %PTL_EXTRA% python -c "pass" >nul 2>&1
if errorlevel 1 set "PTL_OFFLINE="
if not defined PTL_OFFLINE echo Something is missing from what is installed -- fetching it...
if not defined PTL_OFFLINE echo.

REM From the Studio's start to its window's last word is one block, which cmd
REM reads whole before running any of it: an upgrade replaces this file while
REM the block waits, and the block neither sees that nor comes back to the
REM file. Keep it one block, with no "goto" and no "call" to a label in it,
REM and no bracket in what it prints.
(
"%PTL_UV%" run %PTL_OFFLINE% %PTL_EXTRA% panther-lake-launcher %*
if errorlevel 3 if not errorlevel 4 (
echo.
echo Upgrading: the new version starts in a new window once it is installed.
echo This window can be closed.
timeout /t 10 >nul
exit /b 0
)
if errorlevel 1 (
echo.
echo The Studio has stopped.
echo If it did not start at all, the message above says why: when a copy is
echo already running, stop_launcher.bat stops it; to repair the installation,
echo run first_launch.bat.
)
pause
exit /b 0
)

:notinstalled
echo.
echo  The Studio is not installed on this laptop yet.
echo  Double-click first_launch.bat in this folder: it installs everything, step by step.
echo.
pause
