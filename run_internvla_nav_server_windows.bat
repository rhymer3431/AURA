@echo off
setlocal EnableExtensions

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

set "DEFAULT_INTERNVLA_MODEL_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.Q4_K_M.gguf"
set "DEFAULT_INTERNVLA_PYTHON=python"
if exist "%REPO_DIR%\.venv-quant-win\Scripts\python.exe" set "DEFAULT_INTERNVLA_PYTHON=%REPO_DIR%\.venv-quant-win\Scripts\python.exe"

set "DEFAULT_LLAMA_CPP_ROOT=%REPO_DIR%\llama.cpp"
set "DEFAULT_LLAMA_MODEL_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.Q4_K_M.gguf"
set "DEFAULT_LLAMA_MMPROJ_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.mmproj-Q8_0.gguf"
set "DEFAULT_LLAMA_URL=http://127.0.0.1:15802"
set "DEFAULT_LLAMA_CTX_SIZE=8192"
set "DEFAULT_LLAMA_CACHE_TYPE_K=q8_0"
set "DEFAULT_LLAMA_CACHE_TYPE_V=q8_0"

if not defined INTERNVLA_HOST set "INTERNVLA_HOST=127.0.0.1"
if not defined INTERNVLA_PORT set "INTERNVLA_PORT=15801"
if not defined INTERNVLA_BACKEND set "INTERNVLA_BACKEND=llama_cpp"
if not defined INTERNVLA_PYTHON set "INTERNVLA_PYTHON=%DEFAULT_INTERNVLA_PYTHON%"
if not defined INTERNVLA_MODEL_PATH if exist "%DEFAULT_INTERNVLA_MODEL_PATH%" set "INTERNVLA_MODEL_PATH=%DEFAULT_INTERNVLA_MODEL_PATH%"
if not defined INTERNVLA_SKIP_SYSTEM1_TRAJECTORY (
    set "INTERNVLA_SKIP_SYSTEM1_TRAJECTORY=1"
)
if not defined INTERNVLA_INFERENCE_ONLY set "INTERNVLA_INFERENCE_ONLY=1"
if not defined INTERNVLA_SAVE_DEBUG_ARTIFACTS set "INTERNVLA_SAVE_DEBUG_ARTIFACTS=0"
if not defined INTERNVLA_ENABLE_TF32 set "INTERNVLA_ENABLE_TF32=1"
if not defined INTERNVLA_MAX_NEW_TOKENS set "INTERNVLA_MAX_NEW_TOKENS=16"
if not defined INTERNVLA_RESIZE_W set "INTERNVLA_RESIZE_W=384"
if not defined INTERNVLA_RESIZE_H set "INTERNVLA_RESIZE_H=384"
if not defined INTERNVLA_NUM_HISTORY set "INTERNVLA_NUM_HISTORY=4"
if not defined INTERNVLA_PLAN_STEP_GAP set "INTERNVLA_PLAN_STEP_GAP=4"
if not defined INTERNVLA_LLAMA_CPP_ROOT set "INTERNVLA_LLAMA_CPP_ROOT=%DEFAULT_LLAMA_CPP_ROOT%"
if not defined INTERNVLA_LLAMA_MODEL_PATH set "INTERNVLA_LLAMA_MODEL_PATH=%DEFAULT_LLAMA_MODEL_PATH%"
if not defined INTERNVLA_LLAMA_MMPROJ_PATH set "INTERNVLA_LLAMA_MMPROJ_PATH=%DEFAULT_LLAMA_MMPROJ_PATH%"
if not defined INTERNVLA_LLAMA_URL set "INTERNVLA_LLAMA_URL=%DEFAULT_LLAMA_URL%"
if not defined INTERNVLA_LLAMA_CTX_SIZE set "INTERNVLA_LLAMA_CTX_SIZE=%DEFAULT_LLAMA_CTX_SIZE%"
if not defined INTERNVLA_LLAMA_GPU_LAYERS set "INTERNVLA_LLAMA_GPU_LAYERS=all"
if not defined INTERNVLA_LLAMA_MAIN_GPU set "INTERNVLA_LLAMA_MAIN_GPU=0"
if not defined INTERNVLA_LLAMA_CACHE_TYPE_K set "INTERNVLA_LLAMA_CACHE_TYPE_K=%DEFAULT_LLAMA_CACHE_TYPE_K%"
if not defined INTERNVLA_LLAMA_CACHE_TYPE_V set "INTERNVLA_LLAMA_CACHE_TYPE_V=%DEFAULT_LLAMA_CACHE_TYPE_V%"
if not defined INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT set "INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT=You are a binary visual verifier attached to the InternVLA navigation server. Inspect the current image and answer the user's yes-or-no subgoal question. Respond with exactly one lowercase token: true or false. If the image does not clearly support true, respond false."
if not defined INTERNVLA_DEFAULT_CHECK_SESSION_ID set "INTERNVLA_DEFAULT_CHECK_SESSION_ID=check-default"
if not defined INTERNVLA_DEFAULT_CHECK_SESSION_AUTO_OPEN set "INTERNVLA_DEFAULT_CHECK_SESSION_AUTO_OPEN=1"
if not defined INTERNVLA_LLAMA_PARALLEL_SLOTS set "INTERNVLA_LLAMA_PARALLEL_SLOTS=2"
if not defined INTERNVLA_LLAMA_NAV_SLOT set "INTERNVLA_LLAMA_NAV_SLOT=0"
if not defined INTERNVLA_LLAMA_CHECK_SLOT set "INTERNVLA_LLAMA_CHECK_SLOT=1"

set "SERVER_SCRIPT=%REPO_DIR%\serve_internvla_nav_server.py"

if not exist "%SERVER_SCRIPT%" (
    echo [ERROR] InternVLA nav server script not found: %SERVER_SCRIPT%
    exit /b 1
)

if /I not "%INTERNVLA_BACKEND%"=="llama_cpp" (
    echo [ERROR] This launcher only supports INTERNVLA_BACKEND=llama_cpp. Got: %INTERNVLA_BACKEND%
    exit /b 1
)

:validate_llama_cpp
if not exist "%INTERNVLA_LLAMA_CPP_ROOT%" (
    echo [ERROR] INTERNVLA_LLAMA_CPP_ROOT not found: %INTERNVLA_LLAMA_CPP_ROOT%
    exit /b 1
)
if not exist "%INTERNVLA_LLAMA_CPP_ROOT%\llama-server.exe" (
    echo [ERROR] llama-server.exe not found under: %INTERNVLA_LLAMA_CPP_ROOT%
    exit /b 1
)
if not exist "%INTERNVLA_LLAMA_MODEL_PATH%" (
    echo [ERROR] INTERNVLA_LLAMA_MODEL_PATH not found: %INTERNVLA_LLAMA_MODEL_PATH%
    exit /b 1
)
if not exist "%INTERNVLA_LLAMA_MMPROJ_PATH%" (
    echo [ERROR] INTERNVLA_LLAMA_MMPROJ_PATH not found: %INTERNVLA_LLAMA_MMPROJ_PATH%
    exit /b 1
)
echo [INFO] Validating llama.cpp System2 Python environment...
call "%INTERNVLA_PYTHON%" -c "import requests; from PIL import Image" 1>nul 2>nul
if errorlevel 1 (
    echo [ERROR] The selected Python environment cannot run the llama.cpp wrapper server.
    echo [HINT] Required imports: requests, Pillow.
    call "%INTERNVLA_PYTHON%" -c "import requests; print('requests ok'); from PIL import Image; print('Pillow ok')"
    exit /b 1
)
goto :launch

:launch
pushd "%REPO_DIR%"

echo [INFO] Launching InternVLA System2 server...
echo [INFO] Host          : %INTERNVLA_HOST%
echo [INFO] Port          : %INTERNVLA_PORT%
echo [INFO] Backend       : %INTERNVLA_BACKEND%
echo [INFO] Python        : %INTERNVLA_PYTHON%
echo [INFO] Llama root    : %INTERNVLA_LLAMA_CPP_ROOT%
echo [INFO] Llama URL     : %INTERNVLA_LLAMA_URL%
echo [INFO] Llama model   : %INTERNVLA_LLAMA_MODEL_PATH%
echo [INFO] Llama mmproj  : %INTERNVLA_LLAMA_MMPROJ_PATH%
echo [INFO] Llama ctx     : %INTERNVLA_LLAMA_CTX_SIZE%
echo [INFO] Llama layers  : %INTERNVLA_LLAMA_GPU_LAYERS%
echo [INFO] Llama main GPU: %INTERNVLA_LLAMA_MAIN_GPU%
echo [INFO] Llama slots   : %INTERNVLA_LLAMA_PARALLEL_SLOTS%
echo [INFO] Nav slot      : %INTERNVLA_LLAMA_NAV_SLOT%
echo [INFO] Check slot    : %INTERNVLA_LLAMA_CHECK_SLOT%
echo [INFO] Skip System1  : %INTERNVLA_SKIP_SYSTEM1_TRAJECTORY%
echo [INFO] Inference only: %INTERNVLA_INFERENCE_ONLY%
echo [INFO] Save artifacts: %INTERNVLA_SAVE_DEBUG_ARTIFACTS%
echo [INFO] Enable TF32   : %INTERNVLA_ENABLE_TF32%
echo [INFO] Max new tokens: %INTERNVLA_MAX_NEW_TOKENS%
echo [INFO] Resize        : %INTERNVLA_RESIZE_W%x%INTERNVLA_RESIZE_H%
echo [INFO] History       : %INTERNVLA_NUM_HISTORY%
echo [INFO] Plan step gap : %INTERNVLA_PLAN_STEP_GAP%
echo [INFO] Llama KV K    : %INTERNVLA_LLAMA_CACHE_TYPE_K%
echo [INFO] Llama KV V    : %INTERNVLA_LLAMA_CACHE_TYPE_V%
echo [INFO] Check prompt  : %INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT%
echo [INFO] Default check : %INTERNVLA_DEFAULT_CHECK_SESSION_ID%
echo [INFO] Default auto  : %INTERNVLA_DEFAULT_CHECK_SESSION_AUTO_OPEN%
echo [INFO] Extra args    : %*

call "%INTERNVLA_PYTHON%" "%SERVER_SCRIPT%" ^
    --host "%INTERNVLA_HOST%" ^
    --port "%INTERNVLA_PORT%" ^
    --backend "%INTERNVLA_BACKEND%" ^
    --skip-system1-trajectory "%INTERNVLA_SKIP_SYSTEM1_TRAJECTORY%" ^
    --inference-only "%INTERNVLA_INFERENCE_ONLY%" ^
    --save-debug-artifacts "%INTERNVLA_SAVE_DEBUG_ARTIFACTS%" ^
    --enable-tf32 "%INTERNVLA_ENABLE_TF32%" ^
    --max-new-tokens "%INTERNVLA_MAX_NEW_TOKENS%" ^
    --model-path "%INTERNVLA_MODEL_PATH%" ^
    --resize-w "%INTERNVLA_RESIZE_W%" ^
    --resize-h "%INTERNVLA_RESIZE_H%" ^
    --num-history "%INTERNVLA_NUM_HISTORY%" ^
    --plan-step-gap "%INTERNVLA_PLAN_STEP_GAP%" ^
    --llama-cpp-root "%INTERNVLA_LLAMA_CPP_ROOT%" ^
    --llama-model-path "%INTERNVLA_LLAMA_MODEL_PATH%" ^
    --llama-mmproj-path "%INTERNVLA_LLAMA_MMPROJ_PATH%" ^
    --llama-url "%INTERNVLA_LLAMA_URL%" ^
    --llama-ctx-size "%INTERNVLA_LLAMA_CTX_SIZE%" ^
    --llama-gpu-layers "%INTERNVLA_LLAMA_GPU_LAYERS%" ^
    --llama-main-gpu "%INTERNVLA_LLAMA_MAIN_GPU%" ^
    --llama-parallel-slots "%INTERNVLA_LLAMA_PARALLEL_SLOTS%" ^
    --llama-nav-slot "%INTERNVLA_LLAMA_NAV_SLOT%" ^
    --llama-check-slot "%INTERNVLA_LLAMA_CHECK_SLOT%" ^
    --llama-cache-type-k "%INTERNVLA_LLAMA_CACHE_TYPE_K%" ^
    --llama-cache-type-v "%INTERNVLA_LLAMA_CACHE_TYPE_V%" ^
    --check-session-system-prompt "%INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT%" ^
    --default-check-session-id "%INTERNVLA_DEFAULT_CHECK_SESSION_ID%" ^
    --default-check-session-auto-open "%INTERNVLA_DEFAULT_CHECK_SESSION_AUTO_OPEN%" ^
    %*

set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" pause
exit /b %RC%
