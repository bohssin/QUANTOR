# Start the QUANTOR app on Windows.
#
#   .\scripts\run.ps1                      # http://quantor:2026
#   .\scripts\run.ps1 -Port 9000
#   .\scripts\run.ps1 -Host 127.0.0.1      # loopback only
#
# Binds 0.0.0.0:2026 so the machine answers to its own name. For `quantor` to
# resolve, add to C:\Windows\System32\drivers\etc\hosts (as Administrator):
#
#     127.0.0.1   quantor
#
# There is no authentication, because this was built for one person's own
# research. 0.0.0.0 means anyone who can reach the machine can drive it, so on
# an untrusted network pass -Host 127.0.0.1 instead.

param(
    [int]$Port = 2026,
    [string]$BindHost = "0.0.0.0",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }

Set-Location (Join-Path $PSScriptRoot "..")
$Root = (Get-Location).Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Error "No virtualenv found. Run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1"
}
& $VenvPy -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "fastapi/uvicorn are missing from .venv. Re-run scripts\setup.ps1."
}

if (-not $env:QUANTOR_LIBRARY) { $env:QUANTOR_LIBRARY = Join-Path $Root "library" }
New-Item -ItemType Directory -Force -Path $env:QUANTOR_LIBRARY | Out-Null

# UTF-8 so a path like "Téléchargements MEGA" survives the round trip.
$env:PYTHONUTF8 = "1"

Write-Host ""
Write-Host "  QUANTOR"
Write-Host "  library : $env:QUANTOR_LIBRARY"
if ($BindHost -eq "0.0.0.0") {
    Write-Host "  open    : http://quantor:$Port   (or http://localhost:$Port)"
    Write-Host "  note    : reachable from your network - pass -BindHost 127.0.0.1 to keep it local"
} else {
    Write-Host "  open    : http://${BindHost}:$Port"
}
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "app.api.main:app", "--host", $BindHost, "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }
& $VenvPy @uvicornArgs
