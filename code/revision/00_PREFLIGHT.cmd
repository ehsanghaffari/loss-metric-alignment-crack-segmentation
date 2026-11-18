@echo off
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
python "01_train_reviewer_revision.py" --preflight-only
if errorlevel 1 exit /b 1
pause
