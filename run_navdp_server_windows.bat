@echo off
setlocal EnableExtensions

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined NAVDP_PORT set "NAVDP_PORT=8888"
if not defined NAVDP_CHECKPOINT set "NAVDP_CHECKPOINT=%REPO_DIR%\artifacts\models\navdp-cross-modal.ckpt"
set "DEFAULT_NAVDP_PYTHON=python"
if defined CONDA_DEFAULT_ENV if /I not "%CONDA_DEFAULT_ENV%"=="base" if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "DEFAULT_NAVDP_PYTHON=%CONDA_PREFIX%\python.exe"
if /I "%DEFAULT_NAVDP_PYTHON%"=="python" if exist "%USERPROFILE%\anaconda3\envs\fa2-cu130-py312\python.exe" set "DEFAULT_NAVDP_PYTHON=%USERPROFILE%\anaconda3\envs\fa2-cu130-py312\python.exe"
if /I "%DEFAULT_NAVDP_PYTHON%"=="python" if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "DEFAULT_NAVDP_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined NAVDP_PYTHON set "NAVDP_PYTHON=%DEFAULT_NAVDP_PYTHON%"
if not defined NAVDP_DEVICE set "NAVDP_DEVICE=cuda:0"
if not defined NAVDP_SAVE_DEBUG_VIDEO set "NAVDP_SAVE_DEBUG_VIDEO=0"
if not defined NAVDP_DEBUG_VIDEO_DIR set "NAVDP_DEBUG_VIDEO_DIR=%REPO_DIR%\artifacts\navdp_debug"

set "NAVDP_SERVER_DIR=%REPO_DIR%\src\navdp"
set "NAVDP_SERVER_SCRIPT=%NAVDP_SERVER_DIR%\navdp_server.py"

if not exist "%NAVDP_SERVER_SCRIPT%" (
    echo [ERROR] NavDP server script not found: %NAVDP_SERVER_SCRIPT%
    exit /b 1
)

if not exist "%NAVDP_CHECKPOINT%" (
    echo [ERROR] NavDP checkpoint not found: %NAVDP_CHECKPOINT%
    exit /b 1
)

echo [INFO] Validating NavDP server Python environment...
call "%NAVDP_PYTHON%" -c "import torch; import flask; import cv2; from PIL import Image" 1>nul 2>nul
if errorlevel 1 (
    echo [ERROR] Missing required NavDP server dependencies in: %NAVDP_PYTHON%
    echo [HINT] Required packages: torch, flask, opencv-python, Pillow
    call "%NAVDP_PYTHON%" -c "import torch; import flask; import cv2; from PIL import Image"
    exit /b 1
)

if /I "%NAVDP_SAVE_DEBUG_VIDEO%"=="1" (
    call "%NAVDP_PYTHON%" -c "import imageio" 1>nul 2>nul
    if errorlevel 1 (
        echo [ERROR] NAVDP_SAVE_DEBUG_VIDEO=1 requires imageio in the server Python environment.
        call "%NAVDP_PYTHON%" -c "import imageio"
        exit /b 1
    )
)

pushd "%NAVDP_SERVER_DIR%"

echo [INFO] Launching bundled NavDP server...
echo [INFO] Script           : %NAVDP_SERVER_SCRIPT%
echo [INFO] Checkpoint       : %NAVDP_CHECKPOINT%
echo [INFO] Port             : %NAVDP_PORT%
echo [INFO] Python           : %NAVDP_PYTHON%
echo [INFO] Device           : %NAVDP_DEVICE%
echo [INFO] Save debug video : %NAVDP_SAVE_DEBUG_VIDEO%
if /I "%NAVDP_SAVE_DEBUG_VIDEO%"=="1" (
    echo [INFO] Debug video dir  : %NAVDP_DEBUG_VIDEO_DIR%
)
echo [INFO] Extra args       : %*

if /I "%NAVDP_SAVE_DEBUG_VIDEO%"=="1" (
    call "%NAVDP_PYTHON%" "%NAVDP_SERVER_SCRIPT%" ^
        --port "%NAVDP_PORT%" ^
        --checkpoint "%NAVDP_CHECKPOINT%" ^
        --device "%NAVDP_DEVICE%" ^
        --save_debug_video ^
        --debug_video_dir "%NAVDP_DEBUG_VIDEO_DIR%" ^
        %*
) else (
    call "%NAVDP_PYTHON%" "%NAVDP_SERVER_SCRIPT%" ^
        --port "%NAVDP_PORT%" ^
        --checkpoint "%NAVDP_CHECKPOINT%" ^
        --device "%NAVDP_DEVICE%" ^
        %*
)

set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" pause
exit /b %RC%
