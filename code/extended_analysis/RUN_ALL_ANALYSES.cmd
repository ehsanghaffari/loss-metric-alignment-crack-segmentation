@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
python -u "02_score_extension.py" --bootstrap 2000
if errorlevel 1 exit /b 1
python -u "05_summarize_original_all_dataset_geometry.py"
if errorlevel 1 exit /b 1
python -u "06_skeleton_hole_sensitivity.py" --architectures unet --unet-seeds 0 1 2 3 4
if errorlevel 1 exit /b 1
python -u "07_annotation_perturbation_skeleton_preserving.py"
if errorlevel 1 exit /b 1
python -u "08_make_representative_qualitative_cases.py"
if errorlevel 1 exit /b 1
pause
