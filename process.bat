@echo off
setlocal

REM Check if .venv exists
if not exist .venv\ (
    echo "[ERROR] Virtual environment (.venv) not found. Please run setup.bat first."
    pause
    exit /b 1
)

REM Check if input files are provided
if "%~1" == "" (
    echo Usage: process.bat [file_or_dir] [options]
    echo Example: process.bat "C:\MyRecording\Source" --normalize-audio
    pause
    exit /b 0
)

call .venv\Scripts\activate  

echo Running Video Processor...
uv run python --no-asr video_processor.py %*

if errorlevel 1 (
    echo [ERROR] Processing failed.
    pause
    exit /b 1
)

echo Processing complete!
pause
exit /b 0
