$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetScript = Join-Path $ScriptDir "run_internvla_system2.ps1"

if (-not (Test-Path -LiteralPath $TargetScript)) {
    Write-Host "[System2 Optional] launcher not found: `"$TargetScript`""
    exit 1
}

& $TargetScript @args
exit $LASTEXITCODE
