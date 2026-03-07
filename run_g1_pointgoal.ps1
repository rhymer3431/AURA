$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Canonical = Join-Path $ScriptDir "scripts\powershell\run_g1_pointgoal.ps1"
if (-not (Test-Path -LiteralPath $Canonical)) {
    Write-Host "[run_g1_pointgoal] canonical launcher not found: `"$Canonical`""
    exit 1
}

& $Canonical @args
exit $LASTEXITCODE
