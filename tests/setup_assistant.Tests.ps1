# Run with: powershell -NoProfile -File tests/setup_assistant.Tests.ps1
# The setup assistant's rules, without serving a page, installing or downloading
# anything, and without Pester. (pytest runs this file too, on Windows.)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../setup/assistant.ps1')

function Assert($Condition, $Message) {
    if (-not $Condition) { throw $Message }
}
$work = Join-Path ([IO.Path]::GetTempPath()) ('ptl-setup-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    # ---- which laptop this is
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra X7 358H') -eq 'Panther Lake') 'Core Ultra X7 358H is Panther Lake.'
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra 5 325') -eq 'Panther Lake') 'Core Ultra 5 325 is Panther Lake.'
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra 7 258V') -eq 'Lunar Lake') 'Core Ultra 7 258V is Lunar Lake.'
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra 9 288V') -eq 'Lunar Lake') 'Core Ultra 9 288V is Lunar Lake.'
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra 7 155H') -eq 'Other Core Ultra') 'Meteor Lake is another Core Ultra.'
    Assert ((Get-ProcessorFamily 'Intel(R) Core(TM) Ultra 7 265K') -eq 'Other Core Ultra') 'Arrow Lake is another Core Ultra, not Lunar Lake.'
    Assert ((Get-ProcessorFamily 'AMD Ryzen 7 7840U') -eq 'Other') 'Another maker is Other.'

    # ---- the folder the project was put in
    $saved = $env:OneDrive
    $env:OneDrive = 'C:\Users\someone\OneDrive'
    $check = Test-ProjectFolder 'C:\Users\someone\OneDrive\Desktop\ptl-ai-studio-main'
    Assert ($check.state -eq 'warn' -and $check.help -like '*OneDrive*') 'A folder inside OneDrive must be warned about.'
    $env:OneDrive = $saved
    $check = Test-ProjectFolder 'C:\PTL-AI-Studio'
    Assert ($check.state -eq 'ok') 'A short, plain folder is fine.'
    $check = Test-ProjectFolder ('C:\Users\someone\Downloads\ptl-ai-studio-main\ptl-ai-studio-main')
    Assert ($check.state -eq 'ok') 'The folder a ZIP download gives must be accepted: installing there was tried and works.'
    $check = Test-ProjectFolder ('C:\' + ('deep\' * 30) + 'ptl')
    Assert ($check.state -eq 'error' -and $check.help -like '*260*') 'A path the Studio could not open its own files under must stop the installation.'
    $check = Test-ProjectFolder 'C:\Tools & Demos\ptl 100%'
    Assert ($check.state -eq 'warn' -and $check.help -like '*character*') 'Characters that command files trip over must be warned about.'
    $check = Test-ProjectFolder (Join-Path ([IO.Path]::GetTempPath()) 'ptl-ai-studio')
    Assert ($check.state -eq 'warn' -and $check.help -like '*temporary*') 'A temporary folder must be warned about.'

    # ---- commands, as cmd.exe is given them
    $line = ConvertTo-CommandLine @('C:\Program Files (x86)\uv\uv.exe', 'sync', '--locked', '--extra', 'openvino')
    Assert ($line -eq '"C:\Program Files (x86)\uv\uv.exe" sync --locked --extra openvino') "A path with spaces or brackets must be quoted, and nothing else: $line"

    # ---- a failed command, in the words of what to do about it
    $hint = Get-FailureHint 'sync' @('error: invalid peer certificate: UnknownIssuer')
    Assert ($hint.option -eq 'certs' -and $hint.title -like '*certificate*') 'A certificate refusal must offer the retry with Windows certificates.'
    Assert ((Get-FailureHint 'sync' @('failed to write: There is not enough space on the disk. (os error 112)')).title -like '*disk*') 'A full disk must be named.'
    Assert ((Get-FailureHint 'sync' @('error: failed to remove file: The process cannot access the file because it is being used by another process. (os error 32)')).title -like '*in use*') 'A locked file must be named.'
    Assert ((Get-FailureHint 'sync' @('The filename or extension is too long. (os error 206)')).title -like '*too long*') 'A path too long must be named.'
    Assert ((Get-FailureHint 'sync' @('error: Failed to fetch: dns error: No such host is known.')).title -like '*did not go through*') 'A network failure must be named.'
    Assert ((Get-FailureHint 'uv-winget' @('No package found matching input criteria.')).title -like '*installer*') 'A failed installer fetch must say to try the other way.'
    Assert ((Get-FailureHint 'sync' @('something nobody foresaw')).title -eq 'This step failed') 'An unknown failure must still say something.'

    # ---- the end of a log that is being written
    $log = Join-Path $work 'a.log'
    $escape = [string][char]27
    [IO.File]::WriteAllText($log, "Resolved 133 packages`r`nDownloading torch 10%`rDownloading torch 55%`rDownloading torch 90%`r`n$escape[32m + numpy==2.5.2$escape[0m`r`n")
    $tail = @(Get-LogTail $log 14)
    Assert ($tail.Count -eq 3) "Three lines were written, redrawings apart: $($tail -join ' | ')"
    Assert ($tail[1] -eq 'Downloading torch 90%') 'A line that was redrawn must be read as its last drawing.'
    Assert ($tail[2] -eq ' + numpy==2.5.2') "Colour codes must be taken out: '$($tail[2])'"
    Assert (@(Get-LogTail (Join-Path $work 'none.log') 5).Count -eq 0) 'A log that is not there yet has no lines.'

    # ---- what the freshly installed environment says of the chips
    $probe = Join-Path $work 'probe.json'
    [IO.File]::WriteAllText($probe, '{"python":"3.12.10","packages":{"numpy":"2.5.2","openvino":"2026.3.0-1"},"errors":{"onnxruntime":"ImportError: DLL load failed while importing onnxruntime_pybind11_state: The specified module could not be found."},"devices":["CPU","GPU"],"gpus":[{"id":"GPU","name":"Intel(R) Arc(TM) 140V GPU (iGPU)"}],"npu":null,"model_cache":"C:\\Users\\x\\.cache\\huggingface\\hub","model_cache_free_gb":120.5}')
    $read = Read-Probe $probe
    $names = @($read.chips | ForEach-Object { $_.name + ':' + $_.found })
    Assert (($names -join ' ') -eq 'CPU:True GPU:True NPU:False') "The chips must be told apart: $($names -join ' ')"
    Assert ($read.errors.Count -eq 1 -and $read.errors[0].help -like '*Visual C++*') 'A DLL that will not load must point at the Visual C++ runtime.'
    Assert ($read.model_cache_free_gb -eq 120.5) 'The free space where the models go must be passed on.'

    # ---- a command that runs, and what its end makes of its step
    $script:Root = $work
    Initialize-AssistantState 8764 8765
    Assert ($script:Token.Length -eq 48) 'The page must be given a long random token.'
    Start-SetupJob 'sync' 'environment' @($env:ComSpec, '/c', 'echo error: invalid peer certificate: UnknownIssuer& exit 3') $null
    Assert ($script:Steps['environment'].state -eq 'running') 'A started step must say it is running.'
    $refused = $false
    try { Start-SetupJob 'probe' 'chips' @($env:ComSpec, '/c', 'echo no') $null } catch { $refused = $true }
    Assert $refused 'A second command must be refused while one is running.'
    $script:Job.process.WaitForExit()
    Update-Job
    Assert ($script:Steps['environment'].state -eq 'error' -and $script:Job.exit -eq 3) 'A failed command must mark its step as failed.'
    Assert ($script:Job.hint.option -eq 'certs') 'The hint must come from what the command said.'
    Start-SetupJob 'sync' 'environment' @($env:ComSpec, '/c', 'echo Installed 133 packages') $null
    $script:Job.process.WaitForExit()
    Update-Job
    Assert ($script:Steps['environment'].state -eq 'ok') 'A command that ends well must mark its step as done.'
    $state = Get-State
    Assert ($state.job.tail[-1] -eq 'Installed 133 packages') 'The page must be shown the last lines of the command.'
    Assert ($state.job.log -like 'logs\setup-sync-*') "The log must be named relative to the project: $($state.job.log)"
    $json = ConvertTo-Json -InputObject $state -Depth 8 -Compress
    Assert ($json -like '*"steps":[[]{*') 'The state must reach the page as JSON with its lists as lists.'

    # ---- the installer, found where a test says, and only orders the page knows
    $script:Overrides = '{"uv": ""}' | ConvertFrom-Json
    Assert ($null -eq (Find-Uv)) 'A plan that says there is no installer must be believed.'
    $script:Overrides = $null
    $unknown = $false
    try { Invoke-Action 'format-c' $null } catch { $unknown = $true }
    Assert $unknown 'An action the assistant does not know must be refused.'
    $failed = $false
    try { Invoke-Action 'sync' $null } catch { $failed = $_.Exception.Message -like '*installer first*' }
    Assert $failed 'Installing must be refused while there is no installer.'
    Invoke-Action 'mark' ([pscustomobject]@{ step = 'models'; state = 'skipped' })
    Assert ($script:Steps['models'].state -eq 'skipped') 'The page may say the models were skipped.'
    Invoke-Action 'mark' ([pscustomobject]@{ step = 'environment'; state = 'pending' })
    Assert ($script:Steps['environment'].state -eq 'ok') 'The page may not unsettle a step it does not drive.'

    # ---- only the page this script served may give it orders
    function New-FakeRequest($Headers) { return [pscustomobject]@{ Request = [pscustomobject]@{ Headers = $Headers } } }
    Assert (Test-Trusted (New-FakeRequest @{ 'X-Setup-Token' = $script:Token; 'Origin' = 'http://127.0.0.1:8764' })) 'The page itself must be trusted.'
    Assert (-not (Test-Trusted (New-FakeRequest @{ 'X-Setup-Token' = $script:Token; 'Origin' = 'http://example.com' }))) 'A page from elsewhere must be refused, token or not.'
    Assert (-not (Test-Trusted (New-FakeRequest @{ 'Origin' = 'http://127.0.0.1:8764' }))) 'A request without the token must be refused.'
    Assert (-not (Test-Trusted (New-FakeRequest @{ 'X-Setup-Token' = 'guess' }))) 'A wrong token must be refused.'

    # ---- a Studio that answers is taken for this folder's only if it is
    $script:Root = 'C:\PTL-AI-Studio'
    Assert (Test-OwnStudio ([pscustomobject]@{ version = '0.2.91'; root = 'c:\ptl-ai-studio\' })) 'The same folder, however it is written, is this Studio.'
    Assert (-not (Test-OwnStudio ([pscustomobject]@{ version = '0.2.91'; root = 'C:\Users\x\Downloads\ptl-ai-studio-main' }))) 'A Studio from another copy is not this one.'
    Assert (-not (Test-OwnStudio ([pscustomobject]@{ version = '0.2.80' }))) 'A Studio that does not say where it runs from is not taken for this one.'
    Initialize-AssistantState 8764 8765
    $script:UvPath = 'uv'
    $script:Overrides = [pscustomobject]@{ 'studio-answer' = [pscustomobject]@{ version = '0.2.91'; root = 'C:\Users\x\Downloads\ptl-ai-studio-main' } }
    Start-Studio
    Assert ($script:Studio.state -eq 'failed' -and $script:Studio.message -like '*Downloads\ptl-ai-studio-main*') "Another copy's Studio on the port must be said, not borrowed: $($script:Studio.message)"
    $script:Overrides = [pscustomobject]@{ 'studio-answer' = [pscustomobject]@{ version = '0.2.91'; root = 'C:\PTL-AI-Studio' } }
    Start-Studio
    Assert ($script:Studio.state -eq 'running' -and -not $script:Studio.owned) "This folder's Studio, already running, is used as it is."
    $script:Overrides = $null

    Write-Host 'Setup assistant tests passed.' -ForegroundColor Green
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
