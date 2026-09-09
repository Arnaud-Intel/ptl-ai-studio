<#
.SYNOPSIS
  Run the test suite the way CI runs it, before pushing.

.DESCRIPTION
  Half the red builds on this repo have come from the same blind spot: a
  dev machine here runs Windows, Python 3.12 and `uv sync --extra openvino`,
  while CI runs Linux, Python 3.11 and a plain `uv sync`. Code that imports
  an openvino-only module at module scope, or a test that assumes Windows
  path semantics, passes locally and fails there.

  This closes the dependency half of that gap locally: it builds a second
  environment matching CI's -- Python 3.11, no extras -- in `.venv-ci`, and
  runs pytest in it. `.venv` is left completely alone, which matters: a
  plain `uv sync` against the default environment would silently REMOVE the
  openvino extras this machine's demos need.

  The OS half of the gap it cannot close. See CONTRIBUTING.md for the
  handful of rules that covers.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/precheck.ps1
#>
[CmdletBinding()]
param(
    # Re-resolve and reinstall the CI environment even if it already exists.
    [switch]$Refresh
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$env:UV_PROJECT_ENVIRONMENT = '.venv-ci'

Write-Host '==> Building a CI-matching environment (Python 3.11, no extras) in .venv-ci' -ForegroundColor Cyan
$syncArgs = @('sync', '--python', '3.11')
if ($Refresh) { $syncArgs += '--reinstall' }
& uv @syncArgs
if ($LASTEXITCODE -ne 0) { throw "uv sync failed ($LASTEXITCODE)" }

Write-Host '==> Running pytest exactly as the workflow does' -ForegroundColor Cyan
& uv run --python 3.11 pytest --color=no -p no:warnings
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'FAILED under CI conditions -- fix this before pushing.' -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ''
Write-Host 'Passed under CI conditions.' -ForegroundColor Green
