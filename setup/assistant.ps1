# Panther Lake AI Studio - setup assistant.
#
# Started by first_launch.bat. It serves one local web page (setup\index.html)
# that walks through the installation, and does what the page asks: look at
# the laptop, get the installer (uv), install the environment, check the
# chips, start the Studio so that the models can be fetched, and hand over.
#
# Written for a laptop with nothing installed: Windows PowerShell 5.1 and the
# browser that Windows comes with are all it needs. It installs nothing by
# itself and asks before every download. It listens on 127.0.0.1 only, and
# only the page it served can give it orders (a token the page carries).
#
# Exit codes: 0 finished or closed, 9 the page could not be served -- the
# caller then falls back on the console helper (first_launch.ps1).
#
# This file is kept in plain ASCII: Windows PowerShell 5.1 reads a script
# without a byte-order mark as ANSI.
param(
    [int]$Port = 8764,
    [int]$StudioPort = 8765,
    [switch]$NoBrowser,
    # For the tests: a JSON file that replaces facts and commands (see Get-Override).
    [string]$Plan = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$script:Root = Split-Path -Parent $PSScriptRoot
$script:Overrides = $null
$script:Serving = $false
# The longest path, below the project folder, of a file the Studio needs to
# open (measured: 125). Installing itself copes with longer ones.
$script:DeepestNeeded = 125
# What a tool writes to colour its output: taken out of what the page is shown.
$script:AnsiColour = [regex]::Escape([string][char]27) + '\[[0-9;]*[A-Za-z]'

# ------------------------------------------------------------------ small helpers

function Get-Override($Name) {
    # What a test plan says in place of the real thing, or $null.
    if ($null -eq $script:Overrides) { return $null }
    $property = $script:Overrides.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function New-Check($Id, $Label, $Value, $State, $Help) {
    # One line of the "This laptop" step. State: ok, info, warn or error.
    return @{ id = $Id; label = $Label; value = [string]$Value; state = $State; help = [string]$Help }
}

function Get-ProcessorFamily($Name) {
    # Lunar Lake: Core Ultra 200V series ("Ultra 7 258V"). Panther Lake: Core
    # Ultra series 3 ("Ultra X7 358H", "Ultra 5 325").
    if ($Name -match 'Ultra\s+X?\d\s+2\d\dV') { return 'Lunar Lake' }
    if ($Name -match 'Ultra\s+X?\d\s+3\d\d') { return 'Panther Lake' }
    if ($Name -match 'Core\(TM\)\s+Ultra') { return 'Other Core Ultra' }
    return 'Other'
}

function Test-ProjectFolder($Path) {
    # What about the folder itself could get in the way. Returns a check.
    $issues = @()
    $state = 'ok'
    foreach ($cloud in @($env:OneDrive, $env:OneDriveCommercial, $env:OneDriveConsumer)) {
        if ($cloud -and $Path.ToLowerInvariant().StartsWith($cloud.ToLowerInvariant())) {
            $issues += 'It is inside OneDrive, which would try to sync some 30,000 installed files. Move the folder out of OneDrive, for instance to C:\PTL-AI-Studio, and start again.'
            $state = 'warn'
            break
        }
    }
    if ($Path.Length + $script:DeepestNeeded -gt 259) {
        $issues += ("Its path is {0} characters long, and Windows stops at 260 with {1} needed below it. Move the folder somewhere shorter, for instance C:\PTL-AI-Studio, and start again." -f $Path.Length, $script:DeepestNeeded)
        $state = 'error'
    }
    if ($Path -match '[%!&^]') {
        $issues += 'Its path has a character (% ! & or ^) that Windows command files trip over. Rename the folder without it.'
        if ($state -eq 'ok') { $state = 'warn' }
    }
    $temp = [IO.Path]::GetTempPath().TrimEnd('\')
    if ($Path.ToLowerInvariant().StartsWith($temp.ToLowerInvariant())) {
        $issues += 'It is in a temporary folder, which Windows empties. Extract the project somewhere that stays, for instance C:\PTL-AI-Studio.'
        if ($state -eq 'ok') { $state = 'warn' }
    }
    $help = 'The Studio is installed inside this folder (a .venv subfolder, about 2 GB). Nothing is installed elsewhere, except Python itself and the models.'
    if ($issues.Count) { $help = ($issues -join ' ') }
    return (New-Check 'folder' 'Project folder' $Path $state $help)
}

function Get-FreeGb($Path) {
    try {
        $root = [IO.Path]::GetPathRoot($Path)
        $drive = New-Object IO.DriveInfo($root)
        return [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    } catch { return $null }
}

function Get-ModelCachePath {
    if ($env:HF_HUB_CACHE) { return $env:HF_HUB_CACHE }
    if ($env:HF_HOME) { return (Join-Path $env:HF_HOME 'hub') }
    return (Join-Path $env:USERPROFILE '.cache\huggingface\hub')
}

function Get-SystemFacts {
    # What can be known of the laptop before anything is installed.
    $planned = Get-Override 'checks'
    if ($null -ne $planned) {
        $list = @()
        foreach ($item in $planned) { $list += (New-Check $item.id $item.label $item.value $item.state $item.help) }
        return $list
    }
    $checks = @()

    $cpu = ''
    try { $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim() } catch { }
    $family = Get-ProcessorFamily $cpu
    if ($family -eq 'Lunar Lake' -or $family -eq 'Panther Lake') {
        $checks += (New-Check 'processor' 'Processor' "$cpu ($family)" 'ok' 'One of the two families the Studio is made for: a CPU, an Arc integrated GPU and an NPU.')
    } elseif ($family -eq 'Other Core Ultra') {
        $checks += (New-Check 'processor' 'Processor' $cpu 'warn' 'An Intel Core Ultra, but neither Lunar Lake nor Panther Lake: the Studio has not been tried on it. Demos fall back on the CPU where a chip is missing.')
    } else {
        $checks += (New-Check 'processor' 'Processor' $cpu 'warn' 'Not a Lunar Lake or Panther Lake processor. The Studio will install, and its demos will run on the CPU; the GPU and NPU options stay greyed out.')
    }

    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $build = [int]$os.BuildNumber
        $state = 'ok'
        $help = 'Windows 11 is what the hardware panel and the system-sound capture are made for.'
        if ($build -lt 22000) { $state = 'warn'; $help = 'This is older than Windows 11. The demos run; the chip gauges and system-sound capture may not.' }
        $checks += (New-Check 'windows' 'Windows' ("{0} (build {1})" -f $os.Caption.Trim(), $build) $state $help)
        $ram = [math]::Round($os.TotalVisibleMemorySize / 1MB, 0)
        if ($ram -ge 30) {
            $checks += (New-Check 'memory' 'Memory' "$ram GB" 'ok' 'Enough for every demo, including the three that use the 30B coding model (about 17 GB for it).')
        } elseif ($ram -ge 15) {
            $checks += (New-Check 'memory' 'Memory' "$ram GB" 'info' 'Enough for most demos. The three that use the 30B coding model (HTML Creator, Code Review, Page Agent) need about 17 GB for it and may not fit.')
        } else {
            $checks += (New-Check 'memory' 'Memory' "$ram GB" 'warn' 'Little memory for AI models: keep to the small demos (speech, object detection, document Q&A).')
        }
    } catch { }

    $cache = Get-ModelCachePath
    $freeProject = Get-FreeGb $script:Root
    $freeCache = Get-FreeGb $cache
    $free = $freeProject
    if ($null -ne $freeCache -and ($null -eq $free -or $freeCache -lt $free)) { $free = $freeCache }
    if ($null -ne $free) {
        $where = [IO.Path]::GetPathRoot($cache)
        if ($free -ge 60) {
            $checks += (New-Check 'disk' 'Free disk space' "$free GB on $where" 'ok' 'The Studio takes about 2 GB, and every model together nearly 50 GB. You choose which models to fetch.')
        } elseif ($free -ge 8) {
            $checks += (New-Check 'disk' 'Free disk space' "$free GB on $where" 'warn' 'Enough for the Studio (2 GB) and some models, not for all of them (nearly 50 GB). You choose which to fetch in step 5.')
        } else {
            $checks += (New-Check 'disk' 'Free disk space' "$free GB on $where" 'error' 'Too little: the Studio alone takes about 2 GB to install, plus as much while downloading. Free some space and check again.')
        }
    }

    $intel = @()
    try { $intel = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' }) } catch { }
    if ($intel.Count) {
        $named = @()
        $old = $false
        foreach ($gpu in $intel) {
            $date = ''
            if ($gpu.DriverDate) { $date = $gpu.DriverDate.ToString('yyyy-MM-dd'); if ($gpu.DriverDate -lt (Get-Date).AddMonths(-12)) { $old = $true } }
            $named += ("{0}, driver {1} ({2})" -f $gpu.Name, $gpu.DriverVersion, $date)
        }
        if ($old) {
            $checks += (New-Check 'graphics' 'Graphics' ($named -join '; ') 'warn' 'This graphics driver is more than a year old. Update it with the Intel Driver & Support Assistant before installing: the GPU demos need a recent one.')
        } else {
            $checks += (New-Check 'graphics' 'Graphics' ($named -join '; ') 'ok' 'An Intel GPU with a recent driver. Step 4 checks that the AI runtime can really use it.')
        }
    } else {
        $checks += (New-Check 'graphics' 'Graphics' 'No Intel GPU found' 'warn' 'The GPU demos need an Intel Arc GPU. Without one, the Studio runs its demos on the CPU.')
    }

    $npu = $null
    # By its name as a whole word: "%NPU%" alone also finds Windows' "Input Configuration Device".
    try {
        $npu = Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceName LIKE '%AI Boost%' OR DeviceName LIKE '%NPU%' OR DeviceName LIKE '%Neural%'" |
            Where-Object { $_.DeviceName -match 'AI Boost|\bNPU\b|Neural Processing' } | Select-Object -First 1
    } catch { }
    if ($npu) {
        $checks += (New-Check 'npu' 'NPU' ("{0}, driver {1}" -f $npu.DeviceName, $npu.DriverVersion) 'ok' 'The NPU is there with a driver. Step 4 checks that the AI runtime can really use it.')
    } else {
        $checks += (New-Check 'npu' 'NPU' 'Not found' 'warn' 'No NPU driver was found. If this laptop has an NPU (Intel AI Boost), install its driver with the Intel Driver & Support Assistant, restart, and check again. Without it the NPU demos use another chip.')
    }

    $missing = @()
    foreach ($dll in @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')) {
        if (-not (Test-Path (Join-Path $env:WINDIR "System32\$dll"))) { $missing += $dll }
    }
    if ($missing.Count) {
        $checks += (New-Check 'vcruntime' 'Windows components' ('Missing: ' + ($missing -join ', ')) 'warn' 'The Microsoft Visual C++ runtime is not installed, and some AI packages need it. Install "Microsoft Visual C++ Redistributable (x64)" from Microsoft (aka.ms/vs/17/release/vc_redist.x64.exe), then check again. Step 4 will tell if anything is still missing.')
    } else {
        $checks += (New-Check 'vcruntime' 'Windows components' 'Visual C++ runtime present' 'ok' 'What the AI packages expect from Windows is there.')
    }

    $checks += (Test-ProjectFolder $script:Root)

    try {
        $battery = Get-CimInstance Win32_Battery | Select-Object -First 1
        if ($battery -and $battery.BatteryStatus -eq 1) {
            $checks += (New-Check 'power' 'Power' ("On battery, {0}%" -f $battery.EstimatedChargeRemaining) 'info' 'Plug the laptop in: the installation downloads for a while, and the integrated GPU is about a third slower on battery.')
        }
    } catch { }

    try {
        $mine = @(Get-CimInstance Win32_Process -Filter "Name='panther-lake-launcher.exe'" | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.ToLowerInvariant().StartsWith($script:Root.ToLowerInvariant()) })
        if ($mine.Count) {
            $checks += (New-Check 'running' 'Studio' 'Already running from this folder' 'warn' 'Close the Studio (its window, or stop_launcher.bat) before installing: Windows will not replace files that are in use.')
        }
    } catch { }
    return $checks
}

# ------------------------------------------------------------------ the internet

$script:Hosts = @(
    @{ id = 'github'; name = 'github.com'; url = 'https://github.com'; what = 'the installer (uv) and Python' },
    @{ id = 'pypi'; name = 'pypi.org'; url = 'https://pypi.org/simple/pip/'; what = 'the packages' },
    @{ id = 'pytorch'; name = 'download.pytorch.org'; url = 'https://download.pytorch.org/whl/cpu/'; what = 'one of the packages (PyTorch)' },
    @{ id = 'huggingface'; name = 'huggingface.co'; url = 'https://huggingface.co'; what = 'the models' }
)

function Start-NetCheck {
    # Asks the four hosts the installation downloads from, all at once and
    # without waiting: the answers are read by Update-NetCheck.
    $script:Net = @{ pending = $true; started = Get-Date; requests = @(); results = @() }
    $planned = Get-Override 'net'
    if ($null -ne $planned) {
        $script:Net.pending = $false
        foreach ($item in $planned) { $script:Net.results += @{ id = $item.id; name = $item.name; what = $item.what; ok = [bool]$item.ok; detail = [string]$item.detail } }
        return
    }
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }
    foreach ($target in $script:Hosts) {
        try {
            $request = [Net.HttpWebRequest]::Create($target.url)
            $request.Method = 'GET'
            $request.Timeout = 8000
            $request.AllowAutoRedirect = $true
            $request.UserAgent = 'PantherLakeAIStudio-Setup'
            $script:Net.requests += @{ target = $target; request = $request; task = $request.GetResponseAsync() }
        } catch {
            $script:Net.results += @{ id = $target.id; name = $target.name; what = $target.what; ok = $false; detail = $_.Exception.Message }
        }
    }
}

function Update-NetCheck {
    if (-not $script:Net -or -not $script:Net.pending) { return }
    $left = @()
    foreach ($entry in $script:Net.requests) {
        $task = $entry.task
        if (-not $task.IsCompleted) {
            if (((Get-Date) - $script:Net.started).TotalSeconds -gt 12) {
                try { $entry.request.Abort() } catch { }
                $script:Net.results += @{ id = $entry.target.id; name = $entry.target.name; what = $entry.target.what; ok = $false; detail = 'No answer in 12 seconds.' }
            } else { $left += $entry }
            continue
        }
        $ok = $false
        $detail = ''
        if ($task.IsFaulted) {
            $inner = $task.Exception.GetBaseException()
            # An answer that is an error page is still an answer: the host is reachable.
            if ($inner -is [Net.WebException] -and $inner.Response) { $ok = $true; try { $inner.Response.Close() } catch { } }
            else { $detail = $inner.Message }
        } else {
            $ok = $true
            try { $task.Result.Close() } catch { }
        }
        $script:Net.results += @{ id = $entry.target.id; name = $entry.target.name; what = $entry.target.what; ok = $ok; detail = $detail }
    }
    $script:Net.requests = $left
    if (-not $left.Count) { $script:Net.pending = $false }
}

# ------------------------------------------------------------------ uv

function Find-Uv {
    # The installer, wherever it may be: asked for by a test, on the PATH,
    # in this project's own folder, or where winget and uv's own installer
    # put it. A new PATH entry is not seen by a running program, hence the list.
    $planned = Get-Override 'uv'
    if ($null -ne $planned) { if ($planned -and (Test-Path $planned)) { return $planned } else { return $null } }
    $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    foreach ($candidate in @(
            (Join-Path $script:Root '.tools\uv\uv.exe'),
            (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\uv.exe'),
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
            (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'))) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Get-UvVersion($Path) {
    try { return ((& $Path --version) | Select-Object -First 1) } catch { return '' }
}

# ------------------------------------------------------------------ jobs: one long command at a time

function ConvertTo-CommandLine($Tokens) {
    $parts = @()
    foreach ($token in $Tokens) {
        $text = [string]$token
        if ($text -match '[\s"&|<>()^]' -or $text -eq '') { $parts += ('"' + $text.Replace('"', '\"') + '"') } else { $parts += $text }
    }
    return ($parts -join ' ')
}

function Start-SetupJob($Action, $Step, $Tokens, $Environment) {
    # Runs a command with its output in a log file, and comes straight back:
    # the page asks how it is going (Get-State), and Update-Job sees it end.
    if ($script:Job -and $script:Job.running) { throw 'Something is still running. Wait for it, or stop it first.' }
    $logs = Join-Path $script:Root 'logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $log = Join-Path $logs ("setup-{0}-{1}.log" -f $Action, (Get-Date -Format 'yyyyMMdd-HHmmss'))
    $line = (ConvertTo-CommandLine $Tokens) + ' > "' + $log + '" 2>&1'
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $env:ComSpec
    $info.Arguments = '/d /s /c "' + $line + '"'
    $info.WorkingDirectory = $script:Root
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    if ($Environment) { foreach ($key in $Environment.Keys) { $info.EnvironmentVariables[$key] = [string]$Environment[$key] } }
    $process = [Diagnostics.Process]::Start($info)
    $script:Job = @{
        action = $Action; step = $Step; running = $true; exit = $null; log = $log; process = $process
        started = Get-Date; command = (ConvertTo-CommandLine $Tokens); hint = $null
    }
    $script:Steps[$Step].state = 'running'
    $script:Steps[$Step].message = ''
}

function Get-LogTail($Path, $Lines) {
    # The end of a log that is still being written. A tool that redraws one
    # line (a download's progress) is read as its last drawing.
    if (-not $Path -or -not (Test-Path $Path)) { return @() }
    try {
        $stream = New-Object IO.FileStream($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
        try {
            $take = [math]::Min($stream.Length, 24000)
            $null = $stream.Seek(-$take, [IO.SeekOrigin]::End)
            $buffer = New-Object byte[] $take
            $read = $stream.Read($buffer, 0, $take)
            $text = [Text.Encoding]::UTF8.GetString($buffer, 0, $read)
        } finally { $stream.Close() }
    } catch { return @() }
    $kept = @()
    foreach ($row in ($text -split "`n")) {
        $last = (($row -split "`r") | Where-Object { $_.Trim() } | Select-Object -Last 1)
        if ($last) { $kept += ($last -replace $script:AnsiColour, '').TrimEnd() }
    }
    if ($kept.Count -gt $Lines) { $kept = $kept[($kept.Count - $Lines)..($kept.Count - 1)] }
    return $kept
}

function Get-FailureHint($Action, $Tail) {
    # A failed command in the words of what to do about it.
    $text = ($Tail -join "`n")
    if ($text -match '(?i)certificate|UnknownIssuer|invalid peer|self.signed|SSL') {
        return @{ title = 'The connection was refused over a certificate'; option = 'certs'
            text = 'A company network often inspects secure connections with its own certificate, which the installer does not know. Try again using the certificates Windows trusts. If that fails too, your IT team has to allow github.com, pypi.org, files.pythonhosted.org and download.pytorch.org.' }
    }
    if ($text -match '(?i)os error 112|not enough space|No space left|disk full') {
        return @{ title = 'The disk is full'; option = ''
            text = 'Free some space on this drive (the Studio needs about 2 GB, and as much again while downloading), then try again. What was already downloaded is kept.' }
    }
    if ($text -match '(?i)os error 32|being used by another process|os error 5\b|Access is denied') {
        return @{ title = 'A file is in use'; option = ''
            text = 'Windows will not replace a file that a program has open. Close the Studio if it is running (its window, or stop_launcher.bat), wait a few seconds, then try again. An antivirus scanning the new files can cause the same: trying again usually passes.' }
    }
    if ($text -match '(?i)os error 206|too long|os error 3\b') {
        return @{ title = 'A path is too long for Windows'; option = ''
            text = 'Move the project folder somewhere shorter, for instance C:\PTL-AI-Studio, and start first_launch.bat again from there.' }
    }
    if ($text -match '(?i)dns error|No such host|timed out|error sending request|connection (was )?(reset|refused|closed)|failed to fetch|network') {
        return @{ title = 'The download did not go through'; option = 'certs'
            text = 'The network dropped, or something between this laptop and the internet blocked the download. Check the connection (a VPN, a proxy, a guest Wi-Fi with a sign-in page) and try again: what was already downloaded is kept.' }
    }
    if ($text -match '(?i)lockfile|uv\.lock|--locked') {
        return @{ title = 'The project files do not match'; option = ''
            text = 'The list of packages (uv.lock) does not match the project. Download or extract the project again, whole, and start over.' }
    }
    if ($Action -like 'uv-*') {
        return @{ title = 'The installer could not be fetched'; option = ''
            text = 'Try the other way of getting it, or install uv yourself (docs.astral.sh/uv) and press Check again.' }
    }
    return @{ title = 'This step failed'; option = ''
        text = 'The last lines below say why. Trying again is safe: what was already done is kept.' }
}

function Read-Probe($Path) {
    # What probe.py found, as the "chips" step shows it.
    $report = Get-Content -Raw -Path $Path | ConvertFrom-Json
    $devices = @($report.devices)
    $chips = @()
    $chips += @{ id = 'CPU'; name = 'CPU'; found = ($devices -contains 'CPU'); detail = 'Every demo can run here.' }
    $gpus = @($report.gpus)
    if ($gpus.Count) {
        foreach ($gpu in $gpus) { $chips += @{ id = $gpu.id; name = 'GPU'; found = $true; detail = $gpu.name } }
    } else {
        $chips += @{ id = 'GPU'; name = 'GPU'; found = $false; detail = 'The AI runtime found no Intel GPU. Update the graphics driver (Intel Driver & Support Assistant), restart, and check again.' }
    }
    if ($report.npu) { $chips += @{ id = 'NPU'; name = 'NPU'; found = $true; detail = $report.npu } }
    else { $chips += @{ id = 'NPU'; name = 'NPU'; found = $false; detail = 'The AI runtime found no NPU. Install the Intel NPU driver (Intel Driver & Support Assistant), restart, and check again.' } }

    $errors = @()
    foreach ($property in $report.errors.PSObject.Properties) {
        $why = [string]$property.Value
        $help = 'Run the installation step again; if it stays, the log has the detail.'
        if ($why -match '(?i)DLL load failed|specified module could not be found') {
            $help = 'A Windows component is missing: install "Microsoft Visual C++ Redistributable (x64)" from Microsoft (aka.ms/vs/17/release/vc_redist.x64.exe), then check again.'
        }
        $errors += @{ name = $property.Name; error = $why; help = $help }
    }
    $packages = @()
    foreach ($property in $report.packages.PSObject.Properties) { $packages += @{ name = $property.Name; version = [string]$property.Value } }
    return @{ chips = $chips; errors = $errors; packages = $packages; python = [string]$report.python
        model_cache = [string]$report.model_cache; model_cache_free_gb = $report.model_cache_free_gb }
}

function Update-Job {
    # Sees a command end, and says what that makes of its step.
    if (-not $script:Job -or -not $script:Job.running) { return }
    if (-not $script:Job.process.HasExited) { return }
    $job = $script:Job
    $job.running = $false
    $job.exit = $job.process.ExitCode
    $job.seconds = [int]((Get-Date) - $job.started).TotalSeconds
    $step = $script:Steps[$job.step]
    $tail = Get-LogTail $job.log 40
    if ($job.action -like 'uv-*') {
        $script:UvPath = Find-Uv
        if ($script:UvPath) { $step.state = 'ok'; $step.message = (Get-UvVersion $script:UvPath) }
        else { $step.state = 'error'; $job.hint = Get-FailureHint $job.action $tail }
        return
    }
    if ($job.action -eq 'sync') {
        if ($job.exit -eq 0) { $step.state = 'ok'; $step.message = ("Installed in {0} s." -f $job.seconds) }
        else { $step.state = 'error'; $job.hint = Get-FailureHint $job.action $tail }
        return
    }
    if ($job.action -eq 'probe') {
        if (Test-Path $script:ProbeFile) {
            try {
                $script:Probe = Read-Probe $script:ProbeFile
                $missing = @($script:Probe.chips | Where-Object { -not $_.found })
                if ($script:Probe.errors.Count) { $step.state = 'error' }
                elseif ($missing.Count) { $step.state = 'warn' }
                else { $step.state = 'ok' }
                return
            } catch { }
        }
        $step.state = 'error'
        $job.hint = @{ title = 'The check could not run'; option = ''
            text = 'The environment did not answer. Go back to step 3 and install again; the last lines below say what happened.' }
    }
}

function Stop-SetupJob {
    if (-not $script:Job -or -not $script:Job.running) { return }
    try { & taskkill.exe /PID $script:Job.process.Id /T /F | Out-Null } catch { }
    $script:Job.running = $false
    $script:Job.exit = -1
    $script:Steps[$script:Job.step].state = 'pending'
    $script:Steps[$script:Job.step].message = 'Stopped. What was already downloaded is kept.'
}

# ------------------------------------------------------------------ the Studio

function Test-Studio {
    # The Studio that answers on the port, if any: its version, and the
    # folder it was started from.
    $planned = Get-Override 'studio-answer'
    if ($null -ne $planned) { return $planned }
    try {
        $request = [Net.HttpWebRequest]::Create(("http://127.0.0.1:{0}/api/version" -f $script:StudioPort))
        $request.Timeout = 1500
        $request.Proxy = $null  # never through a company proxy: this is the laptop itself
        $response = $request.GetResponse()
        $reader = New-Object IO.StreamReader($response.GetResponseStream())
        $body = $reader.ReadToEnd()
        $response.Close()
        return ($body | ConvertFrom-Json)
    } catch { return $null }
}

function Test-OwnStudio($Answer) {
    # Whether the Studio that answered is this folder's. One started from
    # another copy of the project holds other packages and may be another
    # version: its models list is not this installation's.
    if (-not $Answer -or -not $Answer.root) { return $false }
    return ([string]$Answer.root).TrimEnd('\').ToLowerInvariant() -eq $script:Root.TrimEnd('\').ToLowerInvariant()
}

function Start-Studio {
    # Starts the Studio in this very window, so that closing the window
    # stops it -- unless this folder's Studio is already answering, which is
    # then used. Another program on the port, or a Studio from elsewhere,
    # is said, not borrowed.
    $answer = Test-Studio
    if ($answer) {
        if (Test-OwnStudio $answer) {
            $script:Studio = @{ state = 'running'; owned = $false; message = "Already running (version $($answer.version))."; checked = Get-Date }
        } else {
            $from = 'an earlier version of it, started before this update'
            if ($answer.root) { $from = "the copy in $($answer.root)" }
            $script:Studio = @{ state = 'failed'; owned = $false
                message = "Another Studio is already running on this laptop ($from). Close it -- its black window, or stop_launcher.bat in its folder -- then try again." }
        }
        return
    }
    if ($script:Studio.state -eq 'starting') { return }
    $planned = Get-Override 'studio'
    $info = New-Object Diagnostics.ProcessStartInfo
    if ($null -ne $planned) {
        $info.FileName = $planned[0]
        $info.Arguments = (ConvertTo-CommandLine @($planned | Select-Object -Skip 1))
    } else {
        $info.FileName = $script:UvPath
        $info.Arguments = (ConvertTo-CommandLine @('run', '--no-sync', 'panther-lake-launcher', '--no-browser', '--port', $script:StudioPort))
    }
    $info.WorkingDirectory = $script:Root
    $info.UseShellExecute = $false
    $script:StudioProcess = [Diagnostics.Process]::Start($info)
    $script:Studio = @{ state = 'starting'; owned = $true; message = ''; started = Get-Date }
}

function Update-Studio {
    if ($script:Studio.state -eq 'running') {
        # Still there? Asked every few seconds: closed, it has to be started
        # again before its models can be listed.
        if ($script:Studio.checked -and ((Get-Date) - $script:Studio.checked).TotalSeconds -lt 5) { return }
        $script:Studio.checked = Get-Date
        if ($script:Studio.owned -and $script:StudioProcess -and -not $script:StudioProcess.HasExited) { return }
        if (-not (Test-Studio)) { $script:Studio = @{ state = 'stopped'; owned = $false; message = 'The Studio was closed.' } }
        return
    }
    if ($script:Studio.state -ne 'starting') { return }
    $answer = Test-Studio
    if ($answer -and (Test-OwnStudio $answer)) { $script:Studio.state = 'running'; $script:Studio.checked = Get-Date; return }
    if ($script:StudioProcess -and $script:StudioProcess.HasExited) {
        $script:Studio.state = 'failed'
        $script:Studio.message = ("The Studio stopped as it started (exit code {0}). The window that opened this page says why." -f $script:StudioProcess.ExitCode)
        return
    }
    if (((Get-Date) - $script:Studio.started).TotalSeconds -gt 180) {
        $script:Studio.state = 'failed'
        $script:Studio.message = 'The Studio did not answer in three minutes. The window that opened this page says what it is doing.'
    }
}

function Invoke-StudioProxy($Context, $Path) {
    # The page talks to the Studio through here: a page served from one port
    # may not call another port by itself.
    $request = [Net.HttpWebRequest]::Create(("http://127.0.0.1:{0}{1}" -f $script:StudioPort, $Path))
    $request.Method = $Context.Request.HttpMethod
    # The first look at the models walks the whole cache: half a minute on a full one.
    $request.Timeout = 60000
    $request.Proxy = $null
    if ($Context.Request.HttpMethod -ne 'GET') {
        $reader = New-Object IO.StreamReader($Context.Request.InputStream, [Text.Encoding]::UTF8)
        $payload = [Text.Encoding]::UTF8.GetBytes($reader.ReadToEnd())
        $request.ContentType = 'application/json'
        $request.ContentLength = $payload.Length
        $stream = $request.GetRequestStream()
        $stream.Write($payload, 0, $payload.Length)
        $stream.Close()
    }
    $status = 200
    try { $response = $request.GetResponse() }
    catch [Net.WebException] {
        if (-not $_.Exception.Response) { Send-Json $Context @{ error = 'The Studio is not answering.' } 502; return }
        $response = $_.Exception.Response
    }
    $status = [int]$response.StatusCode
    $reader = New-Object IO.StreamReader($response.GetResponseStream(), [Text.Encoding]::UTF8)
    $body = $reader.ReadToEnd()
    $response.Close()
    Send-Text $Context $body 'application/json; charset=utf-8' $status
}

function New-DesktopShortcut {
    # A shortcut to start_launcher.bat on the desktop. Asked for by the page, never made unasked.
    $desktop = [Environment]::GetFolderPath('Desktop')
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path $desktop 'Panther Lake AI Studio.lnk'))
    $shortcut.TargetPath = (Join-Path $script:Root 'start_launcher.bat')
    $shortcut.WorkingDirectory = $script:Root
    $shortcut.Description = 'Start Panther Lake AI Studio'
    $shortcut.Save()
    return (Join-Path $desktop 'Panther Lake AI Studio.lnk')
}

# ------------------------------------------------------------------ what the page is told, and what it may ask

function Get-State {
    Update-NetCheck
    Update-Job
    Update-Studio

    $system = $script:Steps['system']
    $worst = 'ok'
    foreach ($check in $script:Checks) {
        if ($check.state -eq 'error') { $worst = 'error' }
        elseif ($check.state -eq 'warn' -and $worst -ne 'error') { $worst = 'warn' }
    }
    $net = @()
    if ($script:Net) {
        foreach ($result in $script:Net.results) {
            $net += $result
            if (-not $result.ok -and $worst -ne 'error') { $worst = 'warn' }
        }
        # Without the two the installer itself needs, nothing can be installed.
        $needed = @($script:Net.results | Where-Object { ($_.id -eq 'github' -or $_.id -eq 'pypi') -and -not $_.ok })
        if ($needed.Count -and -not $script:Net.pending) { $worst = 'error' }
    }
    if ($script:Net -and $script:Net.pending) { $system.state = 'running' } else { $system.state = $worst }

    $job = $null
    if ($script:Job) {
        $seconds = $script:Job.seconds
        if ($script:Job.running) { $seconds = [int]((Get-Date) - $script:Job.started).TotalSeconds }
        $job = @{ action = $script:Job.action; step = $script:Job.step; running = $script:Job.running; exit = $script:Job.exit
            seconds = $seconds; tail = @(Get-LogTail $script:Job.log 14); hint = $script:Job.hint
            log = $script:Job.log.Substring($script:Root.Length + 1) }
    }
    $steps = @()
    foreach ($id in $script:StepOrder) { $steps += @{ id = $id; state = $script:Steps[$id].state; message = $script:Steps[$id].message } }
    $versionFile = Join-Path $script:Root 'VERSION'
    $version = ''
    if (Test-Path $versionFile) { $version = (Get-Content -Raw $versionFile).Trim() }
    return @{
        version = $version; root = $script:Root; steps = $steps
        checks = @($script:Checks); net = @($net); net_pending = [bool]($script:Net -and $script:Net.pending)
        uv = @{ path = [string]$script:UvPath; version = [string]$script:Steps['tools'].message
            winget = [bool](Get-Command winget -CommandType Application -ErrorAction SilentlyContinue) }
        environment = @{ present = (Test-Path (Join-Path $script:Root '.venv\Scripts\python.exe')) }
        job = $job; probe = $script:Probe
        studio = @{ state = $script:Studio.state; message = [string]$script:Studio.message
            url = ("http://127.0.0.1:{0}" -f $script:StudioPort) }
        done = $script:Done
    }
}

function Invoke-Action($Name, $Option) {
    # Everything the page may ask for. Nothing here takes a command from the
    # page: it names one of these, and that is all.
    switch ($Name) {
        'recheck' {
            $script:Checks = @(Get-SystemFacts)
            Start-NetCheck
            $script:UvPath = Find-Uv
            if ($script:UvPath) { $script:Steps['tools'].state = 'ok'; $script:Steps['tools'].message = (Get-UvVersion $script:UvPath) }
        }
        'uv-recheck' {
            $script:UvPath = Find-Uv
            if ($script:UvPath) { $script:Steps['tools'].state = 'ok'; $script:Steps['tools'].message = (Get-UvVersion $script:UvPath) }
            else { $script:Steps['tools'].state = 'pending'; $script:Steps['tools'].message = 'Still not found.' }
        }
        'uv-winget' {
            $command = Get-Override 'uv-winget'
            if ($null -eq $command) { $command = @('winget', 'install', '--id', 'astral-sh.uv', '--exact', '--source', 'winget', '--accept-source-agreements', '--accept-package-agreements', '--disable-interactivity') }
            Start-SetupJob 'uv-winget' 'tools' $command $null
        }
        'uv-download' {
            $command = Get-Override 'uv-download'
            if ($null -eq $command) {
                $command = @('powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'get_uv.ps1'), '-Destination', (Join-Path $script:Root '.tools\uv'))
            }
            Start-SetupJob 'uv-download' 'tools' $command $null
        }
        'sync' {
            if (-not $script:UvPath) { throw 'Get the installer first (step 2).' }
            $command = Get-Override 'sync'
            if ($null -eq $command) { $command = @($script:UvPath, 'sync', '--locked', '--extra', 'openvino') }
            $environment = @{ UV_HTTP_TIMEOUT = '120'; NO_COLOR = '1' }
            if ($Option -eq 'certs') { $environment['UV_SYSTEM_CERTS'] = 'true'; $environment['UV_NATIVE_TLS'] = 'true' }
            $script:Probe = $null
            $script:Steps['chips'].state = 'pending'
            Start-SetupJob 'sync' 'environment' $command $environment
        }
        'probe' {
            if (-not $script:UvPath) { throw 'Get the installer first (step 2).' }
            $script:ProbeFile = Join-Path $script:Root 'logs\setup-probe.json'
            Remove-Item -Force $script:ProbeFile -ErrorAction SilentlyContinue
            $command = Get-Override 'probe'
            if ($null -eq $command) { $command = @($script:UvPath, 'run', '--no-sync', 'python', (Join-Path $PSScriptRoot 'probe.py'), $script:ProbeFile) }
            else { $command = @($command) + @($script:ProbeFile) }
            Start-SetupJob 'probe' 'chips' $command @{ NO_COLOR = '1' }
        }
        'cancel' { Stop-SetupJob }
        'studio-start' {
            if (-not $script:UvPath) { throw 'Get the installer first (step 2).' }
            Start-Studio
        }
        'mark' {
            # The page says a step it drives itself (the models) is settled.
            if ($Option.step -eq 'models' -and @('ok', 'skipped', 'pending') -contains $Option.state) {
                $script:Steps['models'].state = $Option.state
            }
        }
        'shortcut' { $script:Shortcut = New-DesktopShortcut }
        'open-studio' {
            if (-not (Get-Override 'no-open')) { Start-Process ("http://127.0.0.1:{0}/" -f $script:StudioPort) }
            $script:Steps['ready'].state = 'ok'
        }
        'finish' { $script:Steps['ready'].state = 'ok'; $script:Done = $true }
        default { throw "Unknown action '$Name'." }
    }
}

# ------------------------------------------------------------------ the web server

function Send-Text($Context, $Text, $Type, $Status) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $Context.Response.StatusCode = $Status
    $Context.Response.ContentType = $Type
    $Context.Response.Headers['Cache-Control'] = 'no-store'
    $Context.Response.ContentLength64 = $bytes.Length
    $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Context.Response.Close()
}

function Send-Json($Context, $Value, $Status) {
    Send-Text $Context (ConvertTo-Json -InputObject $Value -Depth 8 -Compress) 'application/json; charset=utf-8' $Status
}

function Send-File($Context, $Path, $Type) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    $Context.Response.StatusCode = 200
    $Context.Response.ContentType = $Type
    $Context.Response.ContentLength64 = $bytes.Length
    $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Context.Response.Close()
}

function Test-Trusted($Context) {
    # An order is taken from the page this script served, and from nothing
    # else: it carries the token, and a page from elsewhere cannot read it.
    $origin = $Context.Request.Headers['Origin']
    if ($origin -and @(("http://127.0.0.1:{0}" -f $script:Port), ("http://localhost:{0}" -f $script:Port)) -notcontains $origin) { return $false }
    return ($Context.Request.Headers['X-Setup-Token'] -eq $script:Token)
}

function Invoke-Request($Context) {
    $path = $Context.Request.Url.AbsolutePath
    $method = $Context.Request.HttpMethod
    # Only this laptop, under its own name: a web page elsewhere cannot make
    # the browser call this server under a name of its choosing.
    if (@('127.0.0.1', 'localhost') -notcontains $Context.Request.Url.Host) { Send-Json $Context @{ error = 'Not here.' } 403; return }

    if ($method -eq 'GET' -and ($path -eq '/' -or $path -eq '/index.html')) {
        $page = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'index.html'), [Text.Encoding]::UTF8)
        Send-Text $Context $page.Replace('__SETUP_TOKEN__', $script:Token) 'text/html; charset=utf-8' 200
        return
    }
    if ($method -eq 'GET' -and $path -like '/static/*') {
        $name = [IO.Path]::GetFileName($path)
        $file = Join-Path $script:Root ("launcher\src\launcher\static\{0}" -f $name)
        if ($name -match '^[\w.-]+\.png$' -and (Test-Path $file)) { Send-File $Context $file 'image/png' } else { Send-Json $Context @{ error = 'Not found.' } 404 }
        return
    }
    if ($method -eq 'GET' -and $path -eq '/api/state') { Send-Json $Context (Get-State) 200; return }

    if (-not (Test-Trusted $Context)) { Send-Json $Context @{ error = 'This request did not come from the setup page.' } 403; return }
    if ($path -like '/studio/api/models*' -and ($method -eq 'GET' -or $method -eq 'POST')) {
        Invoke-StudioProxy $Context $path.Substring('/studio'.Length)
        return
    }
    if ($method -eq 'POST' -and $path -eq '/api/action') {
        $reader = New-Object IO.StreamReader($Context.Request.InputStream, [Text.Encoding]::UTF8)
        $body = $reader.ReadToEnd()
        $asked = $null
        try { $asked = $body | ConvertFrom-Json } catch { }
        if (-not $asked -or -not $asked.name) { Send-Json $Context @{ error = 'No action named.' } 400; return }
        try {
            Invoke-Action ([string]$asked.name) $asked.option
            Send-Json $Context (Get-State) 200
        } catch {
            Send-Json $Context @{ error = $_.Exception.Message } 409
        }
        return
    }
    Send-Json $Context @{ error = 'Not found.' } 404
}

function Initialize-AssistantState($ListenOn, $StudioOn) {
    # Everything the assistant knows, as it is before anything was looked at.
    $script:Port = $ListenOn
    $script:StudioPort = $StudioOn
    $random = New-Object byte[] 24
    (New-Object Security.Cryptography.RNGCryptoServiceProvider).GetBytes($random)
    $script:Token = ([BitConverter]::ToString($random)).Replace('-', '').ToLowerInvariant()
    $script:StepOrder = @('system', 'tools', 'environment', 'chips', 'models', 'ready')
    $script:Steps = @{}
    foreach ($id in $script:StepOrder) { $script:Steps[$id] = @{ state = 'pending'; message = '' } }
    $script:Checks = @()
    $script:Net = $null
    $script:UvPath = $null
    $script:Job = $null
    $script:Probe = $null
    $script:ProbeFile = Join-Path $script:Root 'logs\setup-probe.json'
    $script:Studio = @{ state = 'stopped'; owned = $false; message = '' }
    $script:StudioProcess = $null
    $script:Done = $false
    $script:Shortcut = ''
}

function Start-Assistant {
    if (-not (Test-Path (Join-Path $script:Root 'uv.lock')) -or -not (Test-Path (Join-Path $PSScriptRoot 'index.html'))) {
        Write-Host 'This folder is not the whole project. Extract or clone all of it, then start first_launch.bat from that folder.' -ForegroundColor Red
        return 1
    }
    if ($Plan) { $script:Overrides = Get-Content -Raw -Path $Plan | ConvertFrom-Json }
    Initialize-AssistantState $Port $StudioPort

    Write-Host 'Panther Lake AI Studio - setup assistant' -ForegroundColor Cyan
    $listener = $null
    foreach ($candidate in $Port..($Port + 9)) {
        try {
            $attempt = New-Object Net.HttpListener
            $attempt.Prefixes.Add("http://127.0.0.1:$candidate/")
            $attempt.Start()
            $listener = $attempt
            $script:Port = $candidate
            break
        } catch { }
    }
    if (-not $listener) {
        Write-Host 'The setup page could not be served on this laptop. Falling back on the helper in this window.' -ForegroundColor Yellow
        return 9
    }
    $address = "http://127.0.0.1:$($script:Port)/"
    try {
        Write-Host 'Looking at this laptop...'
        $script:Checks = @(Get-SystemFacts)
        Start-NetCheck
        $script:UvPath = Find-Uv
        if ($script:UvPath) { $script:Steps['tools'].state = 'ok'; $script:Steps['tools'].message = (Get-UvVersion $script:UvPath) }
        if (Test-Path (Join-Path $script:Root '.venv\Scripts\python.exe')) { $script:Steps['environment'].message = 'Already installed here: install again to bring it up to date.' }

        Write-Host ''
        Write-Host "The assistant is open in your browser: $address" -ForegroundColor Green
        Write-Host 'Keep this window open while it works. If the page was closed, open the address above again.'
        Write-Host 'To use the plain helper in this window instead: first_launch.bat console'
        Write-Host ''
        if (-not $NoBrowser) { try { Start-Process $address } catch { Write-Host "Open this address in a browser: $address" -ForegroundColor Yellow } }

        $script:Serving = $true
        $pending = $listener.GetContextAsync()
        while (-not $script:Done) {
            if ($pending.Wait(300)) {
                $context = $pending.Result
                $pending = $listener.GetContextAsync()
                try { $null = Invoke-Request $context }
                catch {
                    try { $null = Send-Json $context @{ error = $_.Exception.Message } 500 } catch { }
                }
            } else {
                Update-Job
                Update-Studio
            }
        }
    } finally {
        try { $listener.Stop(); $listener.Close() } catch { }
    }

    if ($script:Studio.owned -and $script:StudioProcess -and -not $script:StudioProcess.HasExited) {
        Write-Host ''
        Write-Host ("The Studio is running: http://127.0.0.1:{0}" -f $script:StudioPort) -ForegroundColor Green
        Write-Host 'This window is the Studio now. Close it, or press Ctrl+C, to stop the Studio.'
        Write-Host 'Next time, double-click start_launcher.bat.'
        $script:StudioProcess.WaitForExit()
    } else {
        Write-Host ''
        Write-Host 'Setup finished. Next time, double-click start_launcher.bat.' -ForegroundColor Green
    }
    return 0
}

# Dot-sourcing gives the functions to the tests without serving anything.
if ($MyInvocation.InvocationName -ne '.') {
    $code = 1
    # The last thing it returns is its exit code, whatever else a command inside let out.
    try { $code = @(Start-Assistant)[-1] }
    catch {
        Write-Host $_ -ForegroundColor Red
        # Stopped before the page was up, nothing was done yet: the helper in
        # this window takes over (a laptop whose rules forbid what the page needs).
        if (-not $script:Serving) {
            Write-Host 'The setup page could not be started on this laptop. Falling back on the helper in this window.' -ForegroundColor Yellow
            $code = 9
        }
    }
    exit ([int]$code)
}
