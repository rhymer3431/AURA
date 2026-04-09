@echo off
setlocal EnableExtensions
chcp 65001 >nul

for %%I in ("%~dp0..\..\..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

set "DEFAULT_INTERNVLA_PYTHON=python"
if exist "%REPO_DIR%\.venv-quant-win\Scripts\python.exe" set "DEFAULT_INTERNVLA_PYTHON=%REPO_DIR%\.venv-quant-win\Scripts\python.exe"

if not defined INTERNVLA_PYTHON set "INTERNVLA_PYTHON=%DEFAULT_INTERNVLA_PYTHON%"
if not defined INTERNVLA_SERVER_URL set "INTERNVLA_SERVER_URL=http://127.0.0.1:15801"

set "CHECK_MODULE=systems.inference.api.check_internvla_session"

echo [INFO] InternVLA check client
echo [INFO] Python     : %INTERNVLA_PYTHON%
echo [INFO] Server URL : %INTERNVLA_SERVER_URL%
echo [INFO] Extra args : %*
set "PYTHONPATH=%REPO_DIR%\src;%PYTHONPATH%"

call "%INTERNVLA_PYTHON%" -m %CHECK_MODULE% %*
exit /b %ERRORLEVEL%
