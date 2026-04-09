@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

set "DEFAULT_INTERNVLA_PYTHON=python"
if exist "%REPO_DIR%\.venv-quant-win\Scripts\python.exe" set "DEFAULT_INTERNVLA_PYTHON=%REPO_DIR%\.venv-quant-win\Scripts\python.exe"

if not defined INTERNVLA_PYTHON set "INTERNVLA_PYTHON=%DEFAULT_INTERNVLA_PYTHON%"
if not defined INTERNVLA_SERVER_URL set "INTERNVLA_SERVER_URL=http://127.0.0.1:15801"

set "CHECK_SCRIPT=%REPO_DIR%\check_internvla_session.py"

if not exist "%CHECK_SCRIPT%" (
    echo [ERROR] Check client script not found: %CHECK_SCRIPT%
    exit /b 1
)

echo [INFO] InternVLA check client
echo [INFO] Python     : %INTERNVLA_PYTHON%
echo [INFO] Server URL : %INTERNVLA_SERVER_URL%
echo [INFO] Extra args : %*

call "%INTERNVLA_PYTHON%" "%CHECK_SCRIPT%" %*
exit /b %ERRORLEVEL%
