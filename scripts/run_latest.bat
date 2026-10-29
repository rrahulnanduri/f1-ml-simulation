@echo off
rem Offer the latest commits from GitHub, then open the app. Double-click it, or pin a
rem shortcut to it on the desktop / taskbar. Extra arguments go to f1sim (default: menu).
rem New commits are listed and only applied when you say yes: pulling runs that code (and
rem pip runs the build of any changed dependencies), so never update blind.
cd /d "%~dp0.."

git fetch --quiet || (echo Could not reach GitHub; starting the version already here. & goto run)
set NEW=0
for /f %%n in ('git rev-list --count HEAD..@{u} 2^>nul') do set NEW=%%n
if "%NEW%"=="0" goto run

echo.
echo %NEW% new commit(s) on GitHub:
git log --format="  %%h %%an  %%s" HEAD..@{u}
git diff --quiet HEAD @{u} -- pyproject.toml scripts || (
  echo.
  echo They change pyproject.toml or scripts\ - check these before updating:
  git diff --stat HEAD @{u} -- pyproject.toml scripts
)
echo.
choice /c YN /m "Update to them"
if errorlevel 2 goto run

for /f %%i in ('git rev-parse HEAD') do set OLD=%%i
git merge --ff-only @{u} || echo Could not update; starting the version already here.

rem Source is an editable install, so new code runs as is; only new dependencies need pip.
rem constraints.txt pins every dependency to the versions this project was tested with.
git diff --quiet %OLD% HEAD -- pyproject.toml constraints.txt || .venv\Scripts\python -m pip install -q -c constraints.txt -e ".[fast,es]"

:run
if "%~1"=="" (.venv\Scripts\f1sim menu) else (.venv\Scripts\f1sim %*)
if errorlevel 1 pause
