<#
.SYNOPSIS
  Guided Windows setup, model download and first launch. Double-click first_launch.bat.
#>
[CmdletBinding()]
param()

function Read-Choice($Prompt, $Choices, $Default) {
    while ($true) {
        $answer = (Read-Host "$Prompt [$Default]").Trim().ToUpperInvariant()
        if (-not $answer) { return $Default }
        if ($Choices -contains $answer) { return $answer }
        Write-Host "Please enter one of: $($Choices -join ', ')."
    }
}

function Invoke-UvStep($StepArgs, [switch]$Optional, [string]$InputScript) {
    while ($true) {
        if ($InputScript) { $InputScript | & $script:Uv @StepArgs | Out-Host }
        else { & $script:Uv @StepArgs | Out-Host }
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host 'This step failed. Read the error above before retrying.' -ForegroundColor Yellow
        Write-Host 'Connection/timeouts: check your approved network or VPN and proxy settings.'
        Write-Host 'Certificate errors: ask IT for the approved certificate setup; do not disable TLS checks.'
        Write-Host 'Access denied / 401 / 403: check model access and authentication with your administrator.'
        Write-Host 'Disk full: free space on the cache drive shown above. Keep the cache so downloads can resume.'
        $choices = @('R', 'Q')
        $prompt = 'R = retry, Q = quit'
        if ($Optional) { $choices += 'S'; $prompt += ', S = skip (models may still be missing)' }
        $answer = Read-Choice $prompt $choices 'R'
        if ($answer -eq 'S') { return $false }
        if ($answer -eq 'Q') { throw 'Stopped at your request. Run first_launch.bat again to continue.' }
    }
}

function Update-ToolPath {
    # Installers update the registry, not the PATH of an already-open helper.
    $paths = @($env:PATH, [Environment]::GetEnvironmentVariable('Path', 'User'),
        [Environment]::GetEnvironmentVariable('Path', 'Machine'))
    foreach ($folder in @("$env:USERPROFILE/.deno/bin", "$env:LOCALAPPDATA/Microsoft/WinGet/Links")) {
        if (Test-Path -LiteralPath $folder) { $paths += $folder }
    }
    $env:PATH = (($paths -join ';').Split(';') | Where-Object { $_ } | Select-Object -Unique) -join ';'
}

function Install-YouTubeRuntime {
    $winget = Get-Command winget -CommandType Application -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Host 'Windows winget is unavailable. Install Deno using your company-approved method:'
        Write-Host 'https://docs.deno.com/runtime/getting_started/installation/'
        return $false
    }
    Write-Host 'Installing or updating Deno for YouTube using Windows winget...'
    & $winget.Source install --id DenoLand.Deno --exact --source winget | Out-Host
    $installed = $LASTEXITCODE -eq 0
    if (-not $installed) {
        Write-Host 'The Deno installer did not finish successfully. Check its message above.' -ForegroundColor Yellow
        Write-Host 'If company policy blocks installation, use your approved software service or skip YouTube for now.'
    }
    Update-ToolPath
    return $installed
}

function Invoke-YouTubeSetup {
    Write-Host 'Optional YouTube setup check (no YouTube connection is made):'
    Write-Host 'YouTube needs Deno 2.3+ or Node 22+. Direct cameras and local videos do not.'
    while ($true) {
        Update-ToolPath
        & $script:Uv run --no-sync smart-city-doctor | Out-Host
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host 'YouTube setup is not ready. The CHECK lines above identify what needs attention.' -ForegroundColor Yellow
        Write-Host 'If Deno/Node is missing or too old, choose I to install/update Deno.'
        Write-Host 'If you installed it separately, choose R to refresh the search path and check again.'
        $answer = Read-Choice 'I = install Deno, R = recheck, S = continue without YouTube, Q = quit' @('I', 'R', 'S', 'Q') 'S'
        if ($answer -eq 'S') {
            Write-Host 'Continuing without confirmed YouTube support. Direct cameras, local videos and model preparation remain available.'
            return $false
        }
        if ($answer -eq 'Q') { throw 'Stopped at your request. Run first_launch.bat again to continue.' }
        if ($answer -eq 'I') { $null = Install-YouTubeRuntime }
    }
}

function Start-FirstLaunch {
    $ErrorActionPreference = 'Stop'
    # Native tools report failure through their exit code, including on PowerShell 7.
    $PSNativeCommandUseErrorActionPreference = $false
    Set-Location $PSScriptRoot
    if (-not (Test-Path 'uv.lock') -or -not (Test-Path 'launcher/pyproject.toml')) {
        throw 'Extract or clone the whole project first, then run this helper from that folder.'
    }
    New-Item -ItemType Directory -Force 'logs' | Out-Null
    $log = Join-Path $PSScriptRoot ("logs/first-launch-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Start-Transcript -Path $log | Out-Null
    try {
        Write-Host 'Panther Lake AI Studio - guided first launch' -ForegroundColor Cyan
        Write-Host "Support log: $log"
        Write-Host 'You can rerun this helper. Existing downloads are reused; no cache is deleted.'
        Write-Host 'Close any running Studio before installing. No administrator window is required.'
        if ((Read-Choice 'Ready to begin? Y = continue, Q = quit' @('Y', 'Q') 'Y') -eq 'Q') { return }

        Write-Host "`n[1/5] Find the installer (uv)" -ForegroundColor Cyan
        $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
        $script:Uv = if ($command) { $command.Source } else { Join-Path $env:USERPROFILE '.local/bin/uv.exe' }
        while (-not (Test-Path $script:Uv)) {
            Write-Host 'uv is missing. It installs the project and the required Python version.'
            Write-Host 'Install uv using your company-approved method: https://docs.astral.sh/uv/getting-started/installation/'
            Write-Host 'After installation, reopen this helper if your PATH changed.'
            $answer = Read-Choice 'I = install with Windows winget, R = check again, Q = quit' @('I', 'R', 'Q') 'Q'
            if ($answer -eq 'Q') { throw 'Install uv, then rerun first_launch.bat.' }
            if ($answer -eq 'I') {
                if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
                    Write-Host 'winget is unavailable. Use the installation page above.'
                } else {
                    & winget install --id astral-sh.uv --exact --source winget
                    if ($LASTEXITCODE -ne 0) { Write-Host 'uv installation did not complete.' }
                }
            }
            $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
            if ($command) { $script:Uv = $command.Source }
            # winget exposes a link here; a newly installed PATH is not visible in this process.
            foreach ($candidate in @("$env:USERPROFILE/.local/bin/uv.exe", "$env:LOCALAPPDATA/Microsoft/WinGet/Links/uv.exe")) {
                if (Test-Path $candidate) { $script:Uv = $candidate; break }
            }
        }
        & $script:Uv --version

        Write-Host "`n[2/5] Install Python and the Intel/OpenVINO environment" -ForegroundColor Cyan
        Write-Host 'This can take several minutes and needs internet access. The project lockfile is preserved.'
        $null = Invoke-UvStep @('sync', '--locked', '--extra', 'openvino')

        Write-Host "`n[3/5] Check hardware, cache space and launcher imports" -ForegroundColor Cyan
        $probe = @'
import shutil
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
from pantherlake_ai_core.engine import list_openvino_devices
import launcher.app
cache = Path(HF_HUB_CACHE)
cache.mkdir(parents=True, exist_ok=True)
print('Model cache:', cache)
print('Free on cache drive: %.1f GB' % (shutil.disk_usage(cache).free / 1024**3))
devices = list_openvino_devices()
print('OpenVINO devices:', ', '.join(devices) or 'none')
if not any(d.startswith('GPU') for d in devices) or 'NPU' not in devices:
    print('GPU or NPU missing: check your manufacturer-approved Intel drivers and reboot after updating.')
    print('You can continue with CPU. Availability also depends on the model and device support.')
print('Launcher imports OK. This checks discovery, not model inference.')
'@
        $null = Invoke-UvStep @('run', '--no-sync', 'python', '-') -InputScript $probe
        $null = Invoke-YouTubeSetup

        Write-Host "`n[4/5] Prepare models before opening a demo" -ForegroundColor Cyan
        Write-Host '1 = speech starter (OpenVINO Whisper base), 2 = choose a demo, 3 = all models, S = skip'
        Write-Host 'All models include both engines and a coding model of about 17 GB. Allow extra disk space.'
        $selection = Read-Choice 'Choose a download set' @('1', '2', '3', 'S') '1'
        $prepared = $false
        if ($selection -ne 'S') {
            $modelArgs = @('run', '--no-sync', 'panther-lake-prefetch')
            if ($selection -eq '1') { $modelArgs += 'whisper-base-ov' }
            if ($selection -eq '2') {
                # The catalog is shared with the app; do not maintain another demo/model list here.
                $catalog = 'from pantherlake_ai_core.models import MODELS; print("\n".join(sorted({d for m in MODELS for d in m.demos})))'
                $null = Invoke-UvStep @('run', '--no-sync', 'python', '-') -InputScript $catalog
                do { $demo = (Read-Host 'Enter one demo ID from the list above').Trim() } while (-not $demo)
                $modelArgs += @('--demo', $demo)
            }
            $listed = Invoke-UvStep ($modelArgs + @('--list')) -Optional
            if ($listed -and (Read-Choice 'Download this selection? Y = download, S = skip' @('Y', 'S') 'Y') -eq 'Y') {
                $prepared = Invoke-UvStep $modelArgs -Optional
            }
        }
        if (-not $prepared) { Write-Host 'Models were not confirmed ready. Use Prepare models in the app or rerun this helper.' -ForegroundColor Yellow }

        Write-Host "`n[5/5] First launch" -ForegroundColor Cyan
        Write-Host 'The browser opens at http://127.0.0.1:8765. Keep this window open; Ctrl+C stops the server.'
        Write-Host 'For the speech starter: open Live Speech Translation, choose OpenVINO / base and a microphone.'
        Write-Host 'First model loading/compilation can take minutes even after download. Wait for Running.'
        Write-Host 'If a device fails, try CPU and check logs/events.log. Download success is not an inference test.'
        Write-Host 'If port 8765 is occupied, close the other Studio window or use stop_launcher.bat for that copy.'
        Write-Host 'Later launches: double-click start_launcher.bat. Other models/settings may need more downloads.'
        if ((Read-Choice 'Open the Studio now? Y = launch, Q = finish' @('Y', 'Q') 'Y') -eq 'Y') {
            & $script:Uv run --no-sync panther-lake-launcher
            if ($LASTEXITCODE -eq 3) { Write-Host 'Upgrade handed off to a new window.' }
            elseif ($LASTEXITCODE -ne 0) { throw "Launcher exited with code $LASTEXITCODE. See the error above and $log." }
        }
    } finally {
        Write-Host "Support log: $log (review for private paths or URLs before sharing)."
        Stop-Transcript | Out-Null
    }
}

# Dot-sourcing exposes the functions for isolated tests without installing or launching anything.
if ($MyInvocation.InvocationName -ne '.') {
    try { Start-FirstLaunch } catch { Write-Host $_ -ForegroundColor Red; exit 1 }
}
