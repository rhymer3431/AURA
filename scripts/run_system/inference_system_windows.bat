@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined AURA_PYTHON set "AURA_PYTHON=python"
if not defined INFERENCE_SYSTEM_HOST set "INFERENCE_SYSTEM_HOST=127.0.0.1"
if not defined INFERENCE_SYSTEM_PORT set "INFERENCE_SYSTEM_PORT=15880"
if not defined NAVDP_HOST set "NAVDP_HOST=127.0.0.1"
if not defined NAVDP_PORT set "NAVDP_PORT=18888"
if not defined NAVDP_CHECKPOINT set "NAVDP_CHECKPOINT=%REPO_DIR%\navdp-cross-modal.ckpt"
if not defined NAVDP_DEVICE set "NAVDP_DEVICE=cuda:0"
if not defined SYSTEM2_HOST set "SYSTEM2_HOST=127.0.0.1"
if not defined SYSTEM2_PORT set "SYSTEM2_PORT=15801"
if not defined SYSTEM2_LLAMA_URL set "SYSTEM2_LLAMA_URL=http://127.0.0.1:15802"
if not defined PLANNER_MODEL_HOST set "PLANNER_MODEL_HOST=127.0.0.1"
if not defined PLANNER_MODEL_PORT set "PLANNER_MODEL_PORT=8093"
if not defined PLANNER_MODEL_PATH set "PLANNER_MODEL_PATH=%REPO_DIR%\artifacts\models\Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined PLANNER_LLAMA_SERVER set "PLANNER_LLAMA_SERVER=%REPO_DIR%\llama.cpp\llama-server.exe"

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ inference_system_host = $env:INFERENCE_SYSTEM_HOST; inference_system_port = [int]$env:INFERENCE_SYSTEM_PORT; inference_system_url = ('http://{0}:{1}' -f $env:INFERENCE_SYSTEM_HOST, $env:INFERENCE_SYSTEM_PORT); navdp_url = ('http://{0}:{1}' -f $env:NAVDP_HOST, $env:NAVDP_PORT); system2_url = ('http://{0}:{1}' -f $env:SYSTEM2_HOST, $env:SYSTEM2_PORT); planner_model_url = ('http://{0}:{1}/v1/chat/completions' -f $env:PLANNER_MODEL_HOST, $env:PLANNER_MODEL_PORT); child_processes = @('navdp_model','system2','planner_llm') }; $cfg | ConvertTo-Json -Compress -Depth 5"
    exit /b 0
)

pushd "%REPO_DIR%"
set "PYTHONPATH=%REPO_DIR%\src;%PYTHONPATH%"

call "%AURA_PYTHON%" -m systems.inference.api.serve_inference_system ^
    --host "%INFERENCE_SYSTEM_HOST%" ^
    --port "%INFERENCE_SYSTEM_PORT%" ^
    --navdp-host "%NAVDP_HOST%" ^
    --navdp-port "%NAVDP_PORT%" ^
    --navdp-checkpoint "%NAVDP_CHECKPOINT%" ^
    --navdp-device "%NAVDP_DEVICE%" ^
    --system2-host "%SYSTEM2_HOST%" ^
    --system2-port "%SYSTEM2_PORT%" ^
    --system2-llama-url "%SYSTEM2_LLAMA_URL%" ^
    --planner-host "%PLANNER_MODEL_HOST%" ^
    --planner-port "%PLANNER_MODEL_PORT%" ^
    --planner-model-path "%PLANNER_MODEL_PATH%" ^
    --planner-llama-server "%PLANNER_LLAMA_SERVER%" ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
