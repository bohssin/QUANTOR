@echo off
REM Start QUANTOR. Double-click this, or run it from cmd.exe:
REM
REM     run.cmd                     http://quantor:2026
REM     run.cmd -Port 9000
REM     run.cmd -BindHost 127.0.0.1
REM
REM A .cmd wrapper exists because cmd.exe does not execute .ps1 files — typing
REM `.\scripts\run.ps1` at a `C:\>` prompt silently does nothing, which looks
REM exactly like a server that started and said nothing.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run.ps1" %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
  echo.
  echo The server exited with code %EXITCODE%.
  pause
)
exit /b %EXITCODE%
