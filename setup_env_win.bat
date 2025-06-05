@echo off
set PYTHONUTF8=1
setlocal

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo uv is not installed. Installing uv...
    pip install uv
    if %errorlevel% neq 0 (
        echo Failed to install uv. Exiting.
        exit /b 1
    )
) else (
    echo uv is already installed.
)

REM Check if the .venv folder exists
if exist ".venv\" (
    echo .venv folder already exists.
) else (
    echo .venv folder does not exist.

    REM Create the virtual environment using uv
    echo Creating .venv using uv...
    uv venv .venv --python python3.10
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment using uv.
        exit /b 1
    ) else (
        echo Virtual environment created successfully.
    )
)

REM Activate the virtual env
call .venv\Scripts\activate.bat

REM Install and update setuptools and wheel
uv pip install -U setuptools wheel

REM Install other dependencies
uv pip install -U TTS==0.22.0 soundfile ollama triton-windows accelerate sentencepiece imageio imageio-ffmpeg

uv pip install git+https://github.com/huggingface/diffusers.git 

uv pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

uv pip install torchao

uv pip install -U "transformers<4.50" "numpy==1.22.0"


endlocal
pause
