@echo off
setlocal EnableExtensions
chcp 65001 >nul

if not defined NAV_COMMAND_API_HOST set "NAV_COMMAND_API_HOST=127.0.0.1"
if not defined NAV_COMMAND_API_PORT set "NAV_COMMAND_API_PORT=8892"
set "NAV_COMMAND_API_URL=http://%NAV_COMMAND_API_HOST%:%NAV_COMMAND_API_PORT%/nav/command"

if "%~1"=="" goto :show_status

set "NAV_COMMAND=%~1"
set "NAV_COMMAND_LANGUAGE=%~2"
if not defined NAV_COMMAND_LANGUAGE set "NAV_COMMAND_LANGUAGE=auto"

echo [INFO] Sending runtime nav command to %NAV_COMMAND_API_URL%
echo [INFO] Instruction : %NAV_COMMAND%
echo [INFO] Language    : %NAV_COMMAND_LANGUAGE%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); " ^
  "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); " ^
  "$payload = @{ instruction = $env:NAV_COMMAND; language = $env:NAV_COMMAND_LANGUAGE } | ConvertTo-Json -Compress; " ^
  "$response = Invoke-RestMethod -Method Post -Uri $env:NAV_COMMAND_API_URL -ContentType 'application/json' -Body $payload; " ^
  "$response | ConvertTo-Json -Depth 8"
exit /b %ERRORLEVEL%

:show_status
echo [INFO] Querying runtime nav command status from %NAV_COMMAND_API_URL%
echo [INFO] Usage: %~nx0 "go to the doorway" en
echo [INFO] Usage: %~nx0 "go near the red bin" en

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); " ^
  "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); " ^
  "$response = Invoke-RestMethod -Method Get -Uri $env:NAV_COMMAND_API_URL; " ^
  "$response | ConvertTo-Json -Depth 8"
exit /b %ERRORLEVEL%
