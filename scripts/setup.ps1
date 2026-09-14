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

# This file starts with a UTF-8 BOM, and must keep it. Windows PowerShell 5.1 —
# which `powershell.exe` still is — reads a BOM-less .ps1 using the current ANSI
# code page, so the accented path below was already mangled at PARSE time and no
# output setting could undo it ("TÃ©lÃ©chargements"). The BOM is what tells it the
# file is UTF-8; the console encoding below is what keeps it that way on the way
# out. Both are needed, and setting only the second is what shipped first.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONUTF8 = "1"

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
$TestsFailed = ($LASTEXITCODE -ne 0)
if ($TestsFailed) {
    Write-Host ""
    Write-Host "  !! The test suite did not pass. The doctor below will usually" -ForegroundColor Yellow
    Write-Host "  !! name the same cause. Do not skip past this." -ForegroundColor Yellow
}

Write-Host ""
& $VenvPy quantor_mcp\doctor.py
$Status = $LASTEXITCODE
if ($TestsFailed -and $Status -eq 0) { $Status = 1 }

Write-Host @"

Next:
  1. Open the app:
       run.cmd                           # then http://quantor:2026

     (or from PowerShell:  .\scripts\run.ps1)

     For the name `quantor` to resolve, add this line to
     C:\Windows\System32\drivers\etc\hosts (edit as Administrator):

       127.0.0.1   quantor

     Otherwise use http://localhost:2026 - same thing.

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
