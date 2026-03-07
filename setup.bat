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
    uv venv
)
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [1/3] Installing PyTorch with CUDA 12.4 support...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch with CUDA.
    pause
    exit /b 1
)

echo [2/3] Installing ONNX Runtime GPU for faster NeMo inference...
uv pip install onnxruntime-gpu
if errorlevel 1 (
    echo [WARNING] Failed to install onnxruntime-gpu. Falling back to standard execution.
)

echo [3/3] Installing other dependencies from requirements.txt...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Setup complete! NVIDIA GPU support is enabled.
echo You can now use process.bat to run the tool.
pause
exit /b 0
