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

echo.
echo Setup complete! NVIDIA GPU support is enabled.
echo You can now use process.bat to run the tool.
pause
exit /b 0
