@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"

if not defined ISAAC_SIM_PYTHON set "ISAAC_SIM_PYTHON=C:\isaac-sim\python.bat"

set "ISAAC_PYTHON=%ISAAC_SIM_PYTHON%"
set "ENTRY_MODULE=locomotion.entrypoint"
set "SRC_DIR=%REPO_DIR%\src"
set "POLICY=%REPO_DIR%\artifacts\models\policy.onnx"
if not exist "%POLICY%" set "POLICY=%REPO_DIR%\policy.onnx"
set "ROBOT_USD=%REPO_DIR%\src\locomotion\g1\g1_d455.usd"
set "SCENE_DIR=C:\Users\mango\project\isaac\datasets\InteriorAgent\kujiale_0004"
set "PREFERRED_SCENE=%SCENE_DIR%\kujiale_0004_navila_sanitized.usda"
set "FALLBACK_SCENE=%SCENE_DIR%\kujiale_0004.usda"

if defined G1_POINTGOAL_SCENE_USD (
    set "SCENE_USD=%G1_POINTGOAL_SCENE_USD%"
) else if exist "%PREFERRED_SCENE%" (
    set "SCENE_USD=%PREFERRED_SCENE%"
) else (
    set "SCENE_USD=%FALLBACK_SCENE%"
)

if not exist "%ISAAC_PYTHON%" (
    echo [ERROR] Isaac Sim python launcher not found: %ISAAC_PYTHON%
    echo [HINT] Set ISAAC_SIM_PYTHON to your python.bat path.
    exit /b 1
)

if not exist "%SRC_DIR%" (
    echo [ERROR] src directory not found: %SRC_DIR%
    exit /b 1
)

if not exist "%POLICY%" (
    echo [ERROR] ONNX policy not found: %POLICY%
    exit /b 1
)

if not exist "%ROBOT_USD%" (
    echo [ERROR] G1 USD not found: %ROBOT_USD%
    exit /b 1
)

if not exist "%SCENE_USD%" (
    echo [ERROR] Scene USD not found: %SCENE_USD%
    echo [HINT] Set G1_POINTGOAL_SCENE_USD to override the default scene file.
    exit /b 1
)

pushd "%REPO_DIR%"
set "CONDA_PREFIX="
set "OLD_PYTHONPATH=%PYTHONPATH%"
if defined PYTHONPATH (
    set "PYTHONPATH=%SRC_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%SRC_DIR%"
)

echo [INFO] Launching G1 ONNX policy in InteriorAgent scene...
echo [INFO] Isaac Python: %ISAAC_PYTHON%
echo [INFO] Module : %ENTRY_MODULE%
echo [INFO] Policy : %POLICY%
echo [INFO] G1 USD : %ROBOT_USD%
echo [INFO] Scene  : %SCENE_USD%
echo [INFO] Extra args: %*

call "%ISAAC_PYTHON%" -m %ENTRY_MODULE% ^
    --policy "%POLICY%" ^
    --robot_usd "%ROBOT_USD%" ^
    --scene-usd "%SCENE_USD%" ^
    %*

set "RC=%ERRORLEVEL%"
if defined OLD_PYTHONPATH (
    set "PYTHONPATH=%OLD_PYTHONPATH%"
) else (
    set "PYTHONPATH="
)
popd
exit /b %RC%
