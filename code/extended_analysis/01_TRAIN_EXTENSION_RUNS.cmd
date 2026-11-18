@echo off
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
python "01_train_extension.py" --experiment all
if errorlevel 1 exit /b 1
pause
