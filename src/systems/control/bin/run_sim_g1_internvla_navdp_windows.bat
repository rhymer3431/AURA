@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..\..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined CONDA_ENV_NAME set "CONDA_ENV_NAME=fa2-cu130-py312"
if not defined ISAACSIM_PATH set "ISAACSIM_PATH=C:\isaac-sim"
if not defined NAVDP_URL set "NAVDP_URL=http://127.0.0.1:8888"
if not defined INTERNVLA_URL set "INTERNVLA_URL=http://127.0.0.1:15801"
if not defined ENV_URL set "ENV_URL=/Isaac/Environments/Simple_Warehouse/warehouse.usd"
if not defined NAVDP_FALLBACK set "NAVDP_FALLBACK=heuristic"
if not defined NAVDP_AUTOSTART set "NAVDP_AUTOSTART=1"
if not defined INTERNVLA_AUTOSTART set "INTERNVLA_AUTOSTART=1"
if not defined PLANNER_PORT set "PLANNER_PORT=8093"
if not defined PLANNER_BASE_URL set "PLANNER_BASE_URL=http://127.0.0.1:%PLANNER_PORT%/v1/chat/completions"
if not defined PLANNER_HEALTH_URL set "PLANNER_HEALTH_URL=http://127.0.0.1:%PLANNER_PORT%/health"
if not defined PLANNER_MODEL set "PLANNER_MODEL=Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined PLANNER_TIMEOUT set "PLANNER_TIMEOUT=120"
if not defined PLANNER_AUTOSTART set "PLANNER_AUTOSTART=1"
if not defined PLANNER_SERVER_LAUNCHER set "PLANNER_SERVER_LAUNCHER=%REPO_DIR%\scripts\serve_planner_qwen3_nothink.ps1"
if not defined PLANNER_MODEL_PATH set "PLANNER_MODEL_PATH=%REPO_DIR%\artifacts\models\Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined NAV_INSTRUCTION set "NAV_INSTRUCTION=go to the purple box cart on the right side of the warehouse and stop in front of it."
if not defined NAV_INSTRUCTION_LANGUAGE set "NAV_INSTRUCTION_LANGUAGE=en"
if not defined NAV_COMMAND_API_HOST set "NAV_COMMAND_API_HOST=127.0.0.1"
if not defined NAV_COMMAND_API_PORT set "NAV_COMMAND_API_PORT=8892"
if not defined CAMERA_API_HOST set "CAMERA_API_HOST=127.0.0.1"
if not defined CAMERA_API_PORT set "CAMERA_API_PORT=8891"
if not defined CAMERA_PITCH_DEG set "CAMERA_PITCH_DEG=0.0"

call :resolve_port_from_url "%NAVDP_URL%" NAVDP_PORT 8888
call :resolve_port_from_url "%INTERNVLA_URL%" INTERNVLA_PORT 15801

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do (
    if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
)
if "%PRINT_CONFIG_JSON%"=="1" (
    if not defined CONDA_BAT call :resolve_conda_bat
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$asBool = { param([string]$v,[bool]$d=$true) if ([string]::IsNullOrWhiteSpace($v)) { return $d } return -not @('0','false','no','off').Contains($v.Trim().ToLowerInvariant()) }; $cfg = [ordered]@{ conda_env_name = $env:CONDA_ENV_NAME; conda_bat = $env:CONDA_BAT; isaacsim_path = $env:ISAACSIM_PATH; navdp_url = $env:NAVDP_URL; internvla_url = $env:INTERNVLA_URL; navdp_port = [int]$env:NAVDP_PORT; internvla_port = [int]$env:INTERNVLA_PORT; nav_instruction = $env:NAV_INSTRUCTION; nav_instruction_language = $env:NAV_INSTRUCTION_LANGUAGE; nav_command_api_host = $env:NAV_COMMAND_API_HOST; nav_command_api_port = [int]$env:NAV_COMMAND_API_PORT; camera_api_host = $env:CAMERA_API_HOST; camera_api_port = [int]$env:CAMERA_API_PORT; camera_pitch_deg = [double]$env:CAMERA_PITCH_DEG; navdp_autostart = (& $asBool $env:NAVDP_AUTOSTART); internvla_autostart = (& $asBool $env:INTERNVLA_AUTOSTART); navdp_health_url = ('{0}/healthz' -f $env:NAVDP_URL.TrimEnd('/')); internvla_health_url = ('{0}/healthz' -f $env:INTERNVLA_URL.TrimEnd('/')); runtime_args = @() }; $cfg | ConvertTo-Json -Compress -Depth 5"
    exit /b 0
)

if not defined CONDA_BAT call :resolve_conda_bat
if not defined CONDA_BAT (
    echo [ERROR] Could not find conda.bat required to activate %CONDA_ENV_NAME%.
    echo [HINT] Set CONDA_BAT to your Conda install path, for example:
    echo [HINT]   set CONDA_BAT=%%USERPROFILE%%\miniconda3\condabin\conda.bat
    exit /b 1
)

call "%CONDA_BAT%" activate "%CONDA_ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to activate Conda environment: %CONDA_ENV_NAME%
    echo [HINT] Verify the environment exists: conda env list
    echo [HINT] Conda activation script: %CONDA_BAT%
    exit /b 1
)

set "DEFAULT_INTERNVLA_PYTHON=%CONDA_PREFIX%\python.exe"
set "DEFAULT_INTERNVLA_MODEL_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.Q4_K_M.gguf"
set "DEFAULT_LLAMA_CPP_ROOT=%REPO_DIR%\llama.cpp"
set "DEFAULT_LLAMA_MODEL_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.Q4_K_M.gguf"
set "DEFAULT_LLAMA_MMPROJ_PATH=%REPO_DIR%\artifacts\models\InternVLA-N1-System2.mmproj-Q8_0.gguf"
set "DEFAULT_LLAMA_URL=http://127.0.0.1:15802"
set "DEFAULT_LLAMA_CTX_SIZE=8192"
set "DEFAULT_LLAMA_CACHE_TYPE_K=q8_0"
set "DEFAULT_LLAMA_CACHE_TYPE_V=q8_0"

if not defined NAVDP_PYTHON set "NAVDP_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined INTERNVLA_PYTHON set "INTERNVLA_PYTHON=%DEFAULT_INTERNVLA_PYTHON%"
if not defined INTERNVLA_BACKEND set "INTERNVLA_BACKEND=llama_cpp"
if not defined INTERNVLA_MODEL_PATH if exist "%DEFAULT_INTERNVLA_MODEL_PATH%" set "INTERNVLA_MODEL_PATH=%DEFAULT_INTERNVLA_MODEL_PATH%"
if not defined INTERNVLA_SKIP_SYSTEM1_TRAJECTORY (
    set "INTERNVLA_SKIP_SYSTEM1_TRAJECTORY=1"
)
if not defined INTERNVLA_INFERENCE_ONLY set "INTERNVLA_INFERENCE_ONLY=1"
if not defined INTERNVLA_SAVE_DEBUG_ARTIFACTS set "INTERNVLA_SAVE_DEBUG_ARTIFACTS=0"
if not defined INTERNVLA_ENABLE_TF32 set "INTERNVLA_ENABLE_TF32=1"
if not defined INTERNVLA_MAX_NEW_TOKENS set "INTERNVLA_MAX_NEW_TOKENS=16"
if not defined INTERNVLA_NUM_HISTORY set "INTERNVLA_NUM_HISTORY=4"
if not defined INTERNVLA_LLAMA_CPP_ROOT set "INTERNVLA_LLAMA_CPP_ROOT=%DEFAULT_LLAMA_CPP_ROOT%"
if not defined INTERNVLA_LLAMA_MODEL_PATH set "INTERNVLA_LLAMA_MODEL_PATH=%DEFAULT_LLAMA_MODEL_PATH%"
if not defined INTERNVLA_LLAMA_MMPROJ_PATH set "INTERNVLA_LLAMA_MMPROJ_PATH=%DEFAULT_LLAMA_MMPROJ_PATH%"
if not defined INTERNVLA_LLAMA_URL set "INTERNVLA_LLAMA_URL=%DEFAULT_LLAMA_URL%"
if not defined INTERNVLA_LLAMA_CTX_SIZE set "INTERNVLA_LLAMA_CTX_SIZE=%DEFAULT_LLAMA_CTX_SIZE%"
if not defined INTERNVLA_LLAMA_GPU_LAYERS set "INTERNVLA_LLAMA_GPU_LAYERS=all"
if not defined INTERNVLA_LLAMA_MAIN_GPU set "INTERNVLA_LLAMA_MAIN_GPU=0"
if not defined INTERNVLA_LLAMA_CACHE_TYPE_K set "INTERNVLA_LLAMA_CACHE_TYPE_K=%DEFAULT_LLAMA_CACHE_TYPE_K%"
if not defined INTERNVLA_LLAMA_CACHE_TYPE_V set "INTERNVLA_LLAMA_CACHE_TYPE_V=%DEFAULT_LLAMA_CACHE_TYPE_V%"

set "ISAACSIM_PYTHON=%ISAACSIM_PATH%\python.bat"
set "PLAY_MODULE=systems.control.api.play_g1_internvla_navdp"
set "POLICY=%REPO_DIR%\artifacts\models\g1_policy_fp16.engine"
set "CONFIG_DIR=%REPO_DIR%\tuned\params"
set "ROBOT_USD=%REPO_DIR%\robots\g1\g1_d455.usd"
set "NAVDP_SERVER_LAUNCHER=%REPO_DIR%\src\systems\navigation\bin\run_navdp_server_windows.bat"
set "INTERNVLA_SERVER_LAUNCHER=%REPO_DIR%\src\systems\inference\bin\run_internvla_nav_server_windows.bat"
set "NAVDP_HEALTH_URL=%NAVDP_URL%/healthz"
set "INTERNVLA_HEALTH_URL=%INTERNVLA_URL%/healthz"

if not exist "%ISAACSIM_PYTHON%" (
    echo [ERROR] Isaac Sim python.bat not found: %ISAACSIM_PYTHON%
    echo [HINT] Set ISAACSIM_PATH to your Isaac Sim install directory.
    exit /b 1
)

if not exist "%POLICY%" (
    echo [ERROR] TensorRT policy engine not found: %POLICY%
    exit /b 1
)

if not exist "%CONFIG_DIR%\env.yaml" (
    echo [ERROR] Training env config not found: %CONFIG_DIR%\env.yaml
    exit /b 1
)

if not exist "%ROBOT_USD%" (
    echo [ERROR] G1 USD not found: %ROBOT_USD%
    exit /b 1
)

if /I "%NAVDP_AUTOSTART%"=="1" if not exist "%NAVDP_SERVER_LAUNCHER%" (
    echo [ERROR] NavDP server launcher not found: %NAVDP_SERVER_LAUNCHER%
    exit /b 1
)

if /I "%INTERNVLA_AUTOSTART%"=="1" if not exist "%INTERNVLA_SERVER_LAUNCHER%" (
    echo [ERROR] InternVLA server launcher not found: %INTERNVLA_SERVER_LAUNCHER%
    exit /b 1
)
if /I "%PLANNER_AUTOSTART%"=="1" if not exist "%PLANNER_SERVER_LAUNCHER%" (
    echo [ERROR] Planner server launcher not found: %PLANNER_SERVER_LAUNCHER%
    exit /b 1
)
if /I "%PLANNER_AUTOSTART%"=="1" if not exist "%PLANNER_MODEL_PATH%" (
    echo [WARN] Planner model not found: %PLANNER_MODEL_PATH%
    echo [WARN] Disabling planner autostart and remote planner URL. Runtime will use deterministic planning only.
    set "PLANNER_AUTOSTART=0"
    set "PLANNER_BASE_URL="
)
if /I not "%INTERNVLA_BACKEND%"=="llama_cpp" (
    echo [ERROR] This launcher only supports INTERNVLA_BACKEND=llama_cpp. Got: %INTERNVLA_BACKEND%
    exit /b 1
)

echo [INFO] Validating Isaac Sim Python environment for InternVLA/NavDP mode...
call "%ISAACSIM_PYTHON%" -c "import requests; from PIL import Image" 1>nul 2>nul
if errorlevel 1 (
    echo [ERROR] InternVLA/NavDP mode requires requests and Pillow in the Isaac Sim Python environment.
    call "%ISAACSIM_PYTHON%" -c "import requests; from PIL import Image"
    exit /b 1
)
echo [INFO] Validating Isaac Sim TensorRT runtime...
call "%ISAACSIM_PYTHON%" -c "import importlib.util, sys, tensorrt; ok = (importlib.util.find_spec('cuda') is not None) or (importlib.util.find_spec('cuda.cudart') is not None); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo [ERROR] Isaac Sim Python must provide TensorRT and cuda.cudart for locomotion engine execution.
    call "%ISAACSIM_PYTHON%" -c "import importlib.util, sys, tensorrt; print('cuda spec =', importlib.util.find_spec('cuda')); print('cuda.cudart spec =', importlib.util.find_spec('cuda.cudart')); sys.exit(1)"
    exit /b 1
)

pushd "%REPO_DIR%"
set "PYTHONPATH=%REPO_DIR%\src;%PYTHONPATH%"

echo [INFO] One-click launch: InternVLA server + NavDP server + Isaac runtime
echo [INFO] Conda env      : %CONDA_ENV_NAME%
echo [INFO] Conda python   : %CONDA_PREFIX%\python.exe
echo [INFO] Policy         : %POLICY%
echo [INFO] Config dir     : %CONFIG_DIR%
echo [INFO] G1 USD         : %ROBOT_USD%
echo [INFO] NavDP URL      : %NAVDP_URL%
echo [INFO] InternVLA URL  : %INTERNVLA_URL%
echo [INFO] Planner URL    : %PLANNER_BASE_URL%
echo [INFO] Planner model  : %PLANNER_MODEL%
echo [INFO] InternVLA backend: %INTERNVLA_BACKEND%
echo [INFO] InternVLA model: %INTERNVLA_MODEL_PATH%
echo [INFO] InternVLA py   : %INTERNVLA_PYTHON%
echo [INFO] InternVLA s1   : %INTERNVLA_SKIP_SYSTEM1_TRAJECTORY%
echo [INFO] InternVLA infer: %INTERNVLA_INFERENCE_ONLY%
echo [INFO] InternVLA io   : %INTERNVLA_SAVE_DEBUG_ARTIFACTS%
echo [INFO] InternVLA tf32 : %INTERNVLA_ENABLE_TF32%
echo [INFO] InternVLA tok  : %INTERNVLA_MAX_NEW_TOKENS%
if /I "%INTERNVLA_BACKEND%"=="llama_cpp" (
    echo [INFO] Llama.cpp root : %INTERNVLA_LLAMA_CPP_ROOT%
    echo [INFO] Llama.cpp URL  : %INTERNVLA_LLAMA_URL%
    echo [INFO] Llama.cpp GGUF : %INTERNVLA_LLAMA_MODEL_PATH%
    echo [INFO] Llama.cpp proj : %INTERNVLA_LLAMA_MMPROJ_PATH%
    echo [INFO] Llama.cpp ctx  : %INTERNVLA_LLAMA_CTX_SIZE%
    echo [INFO] Llama KV K     : %INTERNVLA_LLAMA_CACHE_TYPE_K%
    echo [INFO] Llama KV V     : %INTERNVLA_LLAMA_CACHE_TYPE_V%
)
echo [INFO] Scene env URL  : %ENV_URL%
echo [INFO] NavDP port     : %NAVDP_PORT%
echo [INFO] InternVLA port : %INTERNVLA_PORT%
echo [INFO] Instruction    : %NAV_INSTRUCTION%
echo [INFO] Language       : %NAV_INSTRUCTION_LANGUAGE%
echo [INFO] Fallback       : %NAVDP_FALLBACK%
if not "%NAV_COMMAND_API_PORT%"=="0" (
    echo [INFO] Command API    : http://%NAV_COMMAND_API_HOST%:%NAV_COMMAND_API_PORT%/nav/command
    echo [INFO] Command helper : %REPO_DIR%\src\systems\control\bin\send_internvla_nav_command_windows.bat
)
echo [INFO] Camera API     : http://%CAMERA_API_HOST%:%CAMERA_API_PORT%/camera/pitch
echo [INFO] Camera tilt    : start=%CAMERA_PITCH_DEG% deg
echo [INFO] Backend        : TensorRT engine
echo [INFO] Extra args     : %*

call :ensure_navdp_server
if errorlevel 1 (
    set "RC=1"
    goto :finish
)

call :ensure_planner_server
if errorlevel 1 (
    set "RC=1"
    goto :finish
)

call :ensure_internvla_server
if errorlevel 1 (
    set "RC=1"
    goto :finish
)

echo [INFO] Launching Isaac runtime...
call "%ISAACSIM_PYTHON%" -m %PLAY_MODULE% ^
    --policy "%POLICY%" ^
    --config_dir "%CONFIG_DIR%" ^
    --robot_usd "%ROBOT_USD%" ^
    --env_url "%ENV_URL%" ^
    --navdp_url "%NAVDP_URL%" ^
    --internvla_url "%INTERNVLA_URL%" ^
    --planner_base_url "%PLANNER_BASE_URL%" ^
    --planner_model "%PLANNER_MODEL%" ^
    --planner_timeout "%PLANNER_TIMEOUT%" ^
    --navdp_fallback "%NAVDP_FALLBACK%" ^
    --nav_instruction "%NAV_INSTRUCTION%" ^
    --nav_instruction_language "%NAV_INSTRUCTION_LANGUAGE%" ^
    --nav_command_api_host "%NAV_COMMAND_API_HOST%" ^
    --nav_command_api_port "%NAV_COMMAND_API_PORT%" ^
    --camera_api_host "%CAMERA_API_HOST%" ^
    --camera_api_port "%CAMERA_API_PORT%" ^
    --camera_pitch_deg "%CAMERA_PITCH_DEG%" ^
    %*

set "RC=%ERRORLEVEL%"
:finish
popd
if not "%RC%"=="0" pause
exit /b %RC%

:ensure_navdp_server
call :wait_for_http "%NAVDP_HEALTH_URL%" "NavDP server" 1 1 1
if not errorlevel 1 (
    echo [INFO] NavDP server already ready: %NAVDP_HEALTH_URL%
    exit /b 0
)
if /I not "%NAVDP_AUTOSTART%"=="1" (
    echo [ERROR] NavDP server is not reachable and NAVDP_AUTOSTART is disabled.
    echo [HINT] Expected health endpoint: %NAVDP_HEALTH_URL%
    exit /b 1
)
echo [INFO] Starting NavDP server window...
start "NavDP Server" cmd /k ""%NAVDP_SERVER_LAUNCHER%""
call :wait_for_http "%NAVDP_HEALTH_URL%" "NavDP server" 90 2
exit /b %ERRORLEVEL%

:ensure_planner_server
if "%PLANNER_BASE_URL%"=="" exit /b 0
call :wait_for_http "%PLANNER_HEALTH_URL%" "Planner server" 1 1 1
if not errorlevel 1 (
    echo [INFO] Planner server already ready: %PLANNER_HEALTH_URL%
    exit /b 0
)
if /I not "%PLANNER_AUTOSTART%"=="1" (
    echo [WARN] Planner server is not reachable. Runtime will use deterministic planning only.
    set "PLANNER_BASE_URL="
    exit /b 0
)
echo [INFO] Starting Planner server window...
start "Planner Server" powershell -NoProfile -ExecutionPolicy Bypass -File "%PLANNER_SERVER_LAUNCHER%" -Model "%PLANNER_MODEL_PATH%" -Port %PLANNER_PORT%
call :wait_for_http "%PLANNER_HEALTH_URL%" "Planner server" 90 2
if errorlevel 1 (
    echo [WARN] Planner server did not become ready. Runtime will use deterministic planning only.
    set "PLANNER_BASE_URL="
    exit /b 0
)
exit /b 0

:ensure_internvla_server
call :wait_for_http "%INTERNVLA_HEALTH_URL%" "InternVLA server" 1 1 1
if not errorlevel 1 (
    echo [INFO] InternVLA server already ready: %INTERNVLA_HEALTH_URL%
    exit /b 0
)
if /I not "%INTERNVLA_AUTOSTART%"=="1" (
    echo [ERROR] InternVLA server is not reachable and INTERNVLA_AUTOSTART is disabled.
    echo [HINT] Expected health endpoint: %INTERNVLA_HEALTH_URL%
    exit /b 1
)
echo [INFO] Starting InternVLA server window...
start "InternVLA Nav Server" cmd /k ""%INTERNVLA_SERVER_LAUNCHER%""
call :wait_for_http "%INTERNVLA_HEALTH_URL%" "InternVLA server" 120 2
exit /b %ERRORLEVEL%

:wait_for_http
setlocal
set "WAIT_URL=%~1"
set "WAIT_NAME=%~2"
set "WAIT_ATTEMPTS=%~3"
set "WAIT_DELAY=%~4"
set "WAIT_SILENT=%~5"
if not defined WAIT_ATTEMPTS set "WAIT_ATTEMPTS=60"
if not defined WAIT_DELAY set "WAIT_DELAY=2"
for /l %%N in (1,1,%WAIT_ATTEMPTS%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%WAIT_URL%' -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        endlocal & exit /b 0
    )
    if %%N lss %WAIT_ATTEMPTS% (
        timeout /t %WAIT_DELAY% /nobreak >nul
    )
)
if /I not "%WAIT_SILENT%"=="1" echo [ERROR] Timed out waiting for %WAIT_NAME%: %WAIT_URL%
endlocal & exit /b 1

:resolve_port_from_url
setlocal
set "TARGET_URL=%~1"
set "DEFAULT_PORT=%~3"
set "RESOLVED_PORT="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$u = [Uri]'%TARGET_URL%'; if ($u.Port -gt 0) { $u.Port }"`) do set "RESOLVED_PORT=%%I"
if not defined RESOLVED_PORT set "RESOLVED_PORT=%DEFAULT_PORT%"
endlocal & set "%~2=%RESOLVED_PORT%" & exit /b 0

:resolve_conda_bat
if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do (
        if exist "%%~dpI..\condabin\conda.bat" (
            set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
            exit /b 0
        )
        if exist "%%~dpIconda.bat" (
            set "CONDA_BAT=%%~dpIconda.bat"
            exit /b 0
        )
    )
)
for %%I in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%ProgramData%\miniconda3\condabin\conda.bat"
    "%ProgramData%\anaconda3\condabin\conda.bat"
) do (
    if exist "%%~I" (
        set "CONDA_BAT=%%~I"
        exit /b 0
    )
)
exit /b 0
