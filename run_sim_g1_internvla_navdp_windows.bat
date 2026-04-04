@echo off
setlocal EnableExtensions

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "HELPER_SCRIPT=%REPO_DIR%\scripts\run_windows_fullstack.ps1"

if not defined CONDA_ENV_NAME set "CONDA_ENV_NAME=fa2-cu130-py312"
if not defined CONDA_BAT call :resolve_conda_bat
if not defined ISAACSIM_PATH set "ISAACSIM_PATH=C:\isaac-sim"
if not defined NAVDP_URL set "NAVDP_URL=http://127.0.0.1:8888"
if not defined INTERNVLA_URL set "INTERNVLA_URL=http://127.0.0.1:15801"
if not defined NAVDP_AUTOSTART set "NAVDP_AUTOSTART=1"
if not defined INTERNVLA_AUTOSTART set "INTERNVLA_AUTOSTART=1"
if not defined NAV_INSTRUCTION set "NAV_INSTRUCTION=Navigate safely to the target and stop when complete."
if not defined NAV_INSTRUCTION_LANGUAGE set "NAV_INSTRUCTION_LANGUAGE=auto"
if not defined NAV_COMMAND_API_HOST set "NAV_COMMAND_API_HOST=127.0.0.1"
if not defined NAV_COMMAND_API_PORT set "NAV_COMMAND_API_PORT=8892"
if not defined CAMERA_API_HOST set "CAMERA_API_HOST=127.0.0.1"
if not defined CAMERA_API_PORT set "CAMERA_API_PORT=8891"
if not defined CAMERA_PITCH_DEG set "CAMERA_PITCH_DEG=0.0"

if not exist "%HELPER_SCRIPT%" (
    echo [ERROR] Full-stack launcher helper not found: %HELPER_SCRIPT%
    exit /b 1
)

echo [AURA_SYSTEM] source-style Windows launch
echo [AURA_SYSTEM] Conda env     : %CONDA_ENV_NAME%
echo [AURA_SYSTEM] Isaac Sim path: %ISAACSIM_PATH%
echo [AURA_SYSTEM] NavDP URL     : %NAVDP_URL%
echo [AURA_SYSTEM] InternVLA URL : %INTERNVLA_URL%
echo [AURA_SYSTEM] Instruction   : %NAV_INSTRUCTION%

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HELPER_SCRIPT%" %*
exit /b %ERRORLEVEL%

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
    "C:\Users\mango\anaconda3\condabin\conda.bat"
    "%ProgramData%\miniconda3\condabin\conda.bat"
    "%ProgramData%\anaconda3\condabin\conda.bat"
) do (
    if exist "%%~I" (
        set "CONDA_BAT=%%~I"
        exit /b 0
    )
)
exit /b 0
