@echo off
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
python "01_train_extension.py" --preflight-only
if errorlevel 1 exit /b 1
pause
