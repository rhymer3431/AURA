$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Canonical = Join-Path $ScriptDir "run_pipeline.ps1"
if (-not (Test-Path -LiteralPath $Canonical)) {
    Write-Host "[run_g1_pointgoal] canonical launcher not found: `"$Canonical`""
    exit 1
}

Write-Host "[run_g1_pointgoal] deprecated launcher name; forwarding to run_pipeline.ps1"
& $Canonical @args
exit $LASTEXITCODE
