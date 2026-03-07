$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Canonical = Join-Path $ScriptDir "scripts\powershell\run_vlm_dual_server.ps1"
if (-not (Test-Path -LiteralPath $Canonical)) {
    Write-Host "[run_vlm_dual_server] canonical launcher not found: `"$Canonical`""
    exit 1
}

& $Canonical @args
exit $LASTEXITCODE
