# Set up QUANTOR on Windows. PowerShell equivalent of scripts/setup.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Creates a venv, installs the engine + app dependencies, writes a .mcp.json
# that points at THAT interpreter with absolute paths, runs the tests, and
# finishes with the doctor.
#
# The interpreter matters more than it looks: when .mcp.json names a python
# without `mcp` installed, the server exits instantly and Claude Code reports
# only CONNECTION_CLOSED — no traceback, nothing naming the cause.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Root = (Get-Location).Path
$Venv = if ($env:QUANTOR_VENV) { $env:QUANTOR_VENV } else { Join-Path $Root ".venv" }

function Find-Python {
    foreach ($c in @("python3.13", "python3.12", "python3.11", "python", "py")) {
        $exe = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $ok = & $c -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"
            if ($LASTEXITCODE -eq 0) { return $c }
        } catch { }
    }
    Write-Error "Need Python 3.11 or newer on PATH. Install it from python.org and tick 'Add python.exe to PATH'."
}

$Py = Find-Python
Write-Host "==> Python: $Py ($(& $Py --version))"

if (-not (Test-Path $Venv)) {
    Write-Host "==> Creating venv at $Venv"
    & $Py -m venv $Venv
}

$VenvPy = Join-Path $Venv "Scripts\python.exe"

Write-Host "==> Installing dependencies"
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet mcp numpy numba pyarrow pandas pytest
& $VenvPy -m pip install --quiet fastapi "uvicorn[standard]" websockets httpx

Write-Host "==> Writing .mcp.json"
& $VenvPy quantor_mcp\doctor.py --write-config

Write-Host "==> Running tests"
& $VenvPy -m pytest tests\ -q

Write-Host ""
& $VenvPy quantor_mcp\doctor.py
$Status = $LASTEXITCODE

Write-Host @"

Next:
  1. Open the app:
       .\scripts\run.ps1                 # then http://quantor:2026

     For the name `quantor` to resolve, add this line to
     C:\Windows\System32\drivers\etc\hosts (edit as Administrator):

       127.0.0.1   quantor

     Otherwise use http://localhost:2026 — same thing.

  2. Load your tick file on the Data page:

       name             xau
       path             C:\Users\HP\Documents\Téléchargements MEGA\xau.csv
       timeframe        M1
       base timeframe   S1
       GMT offset       3

     S1 is the one that matters for a tick archive: the engine fills on bars,
     so S1 underneath an M1 strategy is what settles which of the stop and the
     target was hit first. Without it the engine assumes the stop.

  3. Sign in to LuxAlgo (free; the token is stored per machine):
       npx -y @luxalgo/mcp login
"@
exit $Status
