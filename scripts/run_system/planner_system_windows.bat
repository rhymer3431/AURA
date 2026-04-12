@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined AURA_PYTHON set "AURA_PYTHON=python"
if not defined PLANNER_SYSTEM_HOST set "PLANNER_SYSTEM_HOST=127.0.0.1"
if not defined PLANNER_SYSTEM_PORT set "PLANNER_SYSTEM_PORT=17881"
if not defined NAVIGATION_SYSTEM_URL set "NAVIGATION_SYSTEM_URL=http://127.0.0.1:17882"
if not defined PLANNER_MODEL_BASE_URL set "PLANNER_MODEL_BASE_URL=http://127.0.0.1:8093/v1/chat/completions"
if not defined PLANNER_MODEL_NAME set "PLANNER_MODEL_NAME=Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined PLANNER_TIMEOUT set "PLANNER_TIMEOUT=120.0"

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ planner_system_host = $env:PLANNER_SYSTEM_HOST; planner_system_port = [int]$env:PLANNER_SYSTEM_PORT; planner_system_url = ('http://{0}:{1}' -f $env:PLANNER_SYSTEM_HOST, $env:PLANNER_SYSTEM_PORT); navigation_system_url = $env:NAVIGATION_SYSTEM_URL; planner_model_base_url = $env:PLANNER_MODEL_BASE_URL; planner_model = $env:PLANNER_MODEL_NAME; planner_timeout = [double]$env:PLANNER_TIMEOUT }; $cfg | ConvertTo-Json -Compress -Depth 5"
    exit /b 0
)

pushd "%REPO_DIR%"
set "PYTHONPATH=%REPO_DIR%\src;%PYTHONPATH%"

call "%AURA_PYTHON%" -m systems.planner.api.serve_planner_system ^
    --host "%PLANNER_SYSTEM_HOST%" ^
    --port "%PLANNER_SYSTEM_PORT%" ^
    --navigation-url "%NAVIGATION_SYSTEM_URL%" ^
    --planner-model-base-url "%PLANNER_MODEL_BASE_URL%" ^
    --planner-model "%PLANNER_MODEL_NAME%" ^
    --planner-timeout "%PLANNER_TIMEOUT%" ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
