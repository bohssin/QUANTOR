@echo off
REM Set up QUANTOR. Double-click this, or run it from cmd.exe:
REM
REM     setup.cmd
REM
REM A .cmd wrapper exists because cmd.exe does not execute .ps1 files — typing
REM `.\scripts\setup.ps1` at a `C:\>` prompt silently does nothing, which looks
REM exactly like a script that ran and printed no output.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set EXITCODE=%ERRORLEVEL%
echo.
if not "%1"=="--no-pause" pause
exit /b %EXITCODE%
