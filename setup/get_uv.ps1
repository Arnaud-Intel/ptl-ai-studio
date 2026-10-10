# Fetches uv -- the tool that installs Python and the Studio's packages -- into
# a folder of this project, for a machine that has neither uv nor winget.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup\get_uv.ps1 -Destination .tools\uv
#
# From uv's own releases on GitHub, checked against the checksum published
# beside it. Nothing is installed outside the destination folder, no setting
# is changed and no administrator right is needed: deleting the folder
# removes it.
param(
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$Source = 'https://github.com/astral-sh/uv/releases/latest/download',
    [string]$Asset = 'uv-x86_64-pc-windows-msvc.zip'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes a download several times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$work = Join-Path ([IO.Path]::GetTempPath()) ('ptl-uv-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    $zip = Join-Path $work $Asset
    Write-Output "Downloading $Asset from $Source ..."
    Invoke-WebRequest -UseBasicParsing -Uri "$Source/$Asset" -OutFile $zip
    Invoke-WebRequest -UseBasicParsing -Uri "$Source/$Asset.sha256" -OutFile "$zip.sha256"

    $expected = ((Get-Content -Raw "$zip.sha256").Trim() -split '\s+')[0]
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash
    if ($expected.Length -ne 64 -or $actual -ne $expected.ToUpperInvariant()) {
        throw "The download does not match its published checksum (expected $expected, got $actual). Nothing was installed."
    }
    Write-Output 'Checksum matches the one published with the release.'

    $unpacked = Join-Path $work 'unpacked'
    Expand-Archive -Path $zip -DestinationPath $unpacked
    $found = Get-ChildItem -Path $unpacked -Recurse -Filter 'uv.exe' | Select-Object -First 1
    if (-not $found) { throw 'The archive did not contain uv.exe.' }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -Path $found.DirectoryName -Filter '*.exe' | Copy-Item -Destination $Destination -Force
    $installed = Join-Path $Destination 'uv.exe'
    Write-Output ("Installed: " + (& $installed --version))
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
