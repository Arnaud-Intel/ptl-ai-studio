# Run with: powershell -NoProfile -File tests/first_launch.Tests.ps1
# No installer, network, model or Pester dependency is needed.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../first_launch.ps1')

function Assert($Condition, $Message) {
    if (-not $Condition) { throw $Message }
}
function Read-Host($Prompt) { return $script:Answers.Dequeue() }
function Invoke-FakeUv {
    $script:Calls++
    $script:ReceivedArgs = @($args)
    Write-Output 'Native tool progress must stay visible, not become a return value.'
    $global:LASTEXITCODE = $script:ExitCodes.Dequeue()
}
function Set-Scenario($Codes, $Replies) {
    $script:Calls = 0
    $script:ExitCodes = [System.Collections.Generic.Queue[int]]::new()
    foreach ($code in $Codes) { $script:ExitCodes.Enqueue($code) }
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($reply in $Replies) { $script:Answers.Enqueue($reply) }
    $script:Uv = 'Invoke-FakeUv'
}

Set-Scenario @(0) @()
$result = Invoke-UvStep @('sync', '--locked', '--extra', 'openvino')
Assert ($result -is [bool] -and $result) 'Success must return only true.'
Assert (($script:ReceivedArgs -join ' ') -eq 'sync --locked --extra openvino') 'Installer arguments lost.'

Set-Scenario @(1, 0) @('invalid', 'r')
$result = Invoke-UvStep @('run', '--no-sync', 'panther-lake-prefetch', '--demo', 'screen-ocr')
Assert ($result -and $script:Calls -eq 2) 'Failure must retry the same command.'
Assert ($script:ReceivedArgs[-1] -eq 'screen-ocr') 'Retry lost the demo selection.'

Set-Scenario @(1) @('S')
$result = Invoke-UvStep @('run', '--no-sync', 'panther-lake-prefetch') -Optional
Assert ($result -is [bool] -and -not $result) 'Skipped download must return only false.'

Set-Scenario @(1) @('S', 'Q')
$stopped = $false
try { $null = Invoke-UvStep @('sync') } catch { $stopped = $_.Exception.Message -like 'Stopped at your request*' }
Assert $stopped 'Required setup must reject skip and stop on quit.'

Set-Scenario @() @('')
Assert ((Read-Choice 'Choose' @('1', 'S') '1') -eq '1') 'Enter must select the displayed default.'
function Update-ToolPath { $script:PathRefreshes++ }
function Install-YouTubeRuntime { $script:Installs++; return $script:InstallResult }
$script:PathRefreshes = 0
$script:Installs = 0
$script:InstallResult = $true

Set-Scenario @(1, 0) @('I')
$result = Invoke-YouTubeSetup
Assert ($result -is [bool] -and $result) 'Runtime installation must be followed by a successful doctor check.'
Assert ($script:Installs -eq 1 -and $script:Calls -eq 2) 'Missing runtime must offer installation, then recheck.'
Assert ($script:PathRefreshes -eq 2) 'Each recheck must refresh PATH.'
Assert (($script:ReceivedArgs -join ' ') -eq 'run --no-sync smart-city-doctor') 'Must use the installed doctor without syncing.'

Set-Scenario @(1) @('')
$result = Invoke-YouTubeSetup
Assert ($result -is [bool] -and -not $result) 'Enter must continue without YouTube, not repeat a failed check.'
Assert ($script:Installs -eq 1) 'Skip must not install anything.'

$script:InstallResult = $false
Set-Scenario @(1, 1) @('I', 'S')
$result = Invoke-YouTubeSetup
Assert (-not $result -and $script:Calls -eq 2) 'A failed installation must not claim YouTube is ready.'

Set-Scenario @(1, 0) @('R')
Assert (Invoke-YouTubeSetup) 'A manual install must be picked up by recheck.'

Set-Scenario @(1) @('Q')
$stopped = $false
try { $null = Invoke-YouTubeSetup } catch { $stopped = $_.Exception.Message -like 'Stopped at your request*' }
Assert $stopped 'Quit must stop the helper during YouTube setup.'
Write-Host 'First-launch helper tests passed.' -ForegroundColor Green
