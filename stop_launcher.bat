@echo off
title Panther Lake AI Studio - stop
cd /d "%~dp0"

REM Stops the launcher when its console window is gone (or was never yours).
REM Pass a port to stop a copy running somewhere else: stop_launcher.bat 8766
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"

echo Stopping Panther Lake AI Studio on port %PORT%...
echo.

REM Kills whatever is listening, then waits for the port to actually come
REM free rather than sleeping a fixed guess -- Windows takes a moment to
REM release the socket after the process dies, and a launcher run is two
REM processes, so a second holder can surface after the first one goes.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=%PORT%; $deadline=(Get-Date).AddSeconds(8); $killed=@(); do { $listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; if (-not $listening) { break }; foreach ($id in ($listening.OwningProcess | Select-Object -Unique)) { if ($killed -notcontains $id) { try { $name = (Get-Process -Id $id -ErrorAction Stop).ProcessName; Stop-Process -Id $id -Force -ErrorAction Stop; $killed += $id; Write-Host ('Stopped ' + $name + ' (pid ' + $id + ').') } catch { Write-Host ('Could not stop pid ' + $id + ': ' + $_.Exception.Message) } } }; Start-Sleep -Milliseconds 300 } while ((Get-Date) -lt $deadline); if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { Write-Host ('Port ' + $port + ' is STILL in use -- something would not stop.'); exit 1 }; if ($killed.Count -eq 0) { Write-Host ('Nothing was running on port ' + $port + '.') } else { Write-Host ('Port ' + $port + ' is free.') }"

echo.
pause
