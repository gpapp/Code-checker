@echo off
setlocal
echo Setting up Video Processing Tool with NVIDIA GPU support (RTX 3060)...

REM Check if uv is installed
where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed. Please install it from https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo Creating/Ensuring virtual environment...
if not exist .venv\ (
    uv venv --python 3.13
)
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate  

echo [1/2] Installing other dependencies from requirements.txt...
uv pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [2/2] Installing PyTorch with CUDA 12.6 support...
uv pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch with CUDA.
    pause
    exit /b 1
)

echo [3/3] Checking for PodcastFillerLib model...
if not exist filler_detector.pth (
    if exist PodcastFillerDataset\PodcastFillers.csv (
        if exist fillers_hun\ZO_Hungarian.csv (
            echo [INFO] filler_detector.pth missing. Starting training on BOTH English and Hungarian datasets with Hungarian bias...
            python PodcastFillerLib.py --mode train --csv PodcastFillerDataset\PodcastFillers.csv --clips_dir PodcastFillerDataset\clip_wav --hun_csv fillers_hun\ZO_Hungarian.csv --hun_clips_dir fillers_hun\zo_clips --hun_weight 5.0 --batch_size 256
        ) else (
            echo [INFO] filler_detector.pth missing. Starting training on English dataset only...
            python PodcastFillerLib.py --mode train --csv PodcastFillerDataset\PodcastFillers.csv --clips_dir PodcastFillerDataset\clip_wav --batch_size 256
        )
    ) else (
        echo [SKIP] Trained model missing AND PodcastFillerDataset not found. 
        echo        Download Fillers from: https://zenodo.org/records/7121457
        echo        Unpack to .\PodcastFillerDataset\ before running setup again.
    )
) else (
    echo [OK] filler_detector.pth exists.
)

echo.
echo Setup complete! NVIDIA GPU support is enabled.
echo You can now use process.bat to run the tool.
pause
exit /b 0
