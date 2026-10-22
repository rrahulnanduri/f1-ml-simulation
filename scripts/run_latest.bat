@echo off
rem Update to the latest commit on GitHub, then open the app. Double-click it, or pin a
rem shortcut to it on the desktop / taskbar. Extra arguments go to f1sim (default: menu).
cd /d "%~dp0.."

for /f %%i in ('git rev-parse HEAD') do set OLD=%%i
git pull --ff-only || echo Could not update; starting the version already here.

rem Source is an editable install, so new code runs as is; only new dependencies need pip
git diff --quiet %OLD% HEAD -- pyproject.toml || .venv\Scripts\python -m pip install -q -e ".[fast,es]"

if "%~1"=="" (.venv\Scripts\f1sim menu) else (.venv\Scripts\f1sim %*)
if errorlevel 1 pause
