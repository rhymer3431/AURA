$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LegacyScript = Join-Path $ScriptDir "legacy\run_vlm_dual_server.ps1"

if (-not (Test-Path -LiteralPath $LegacyScript)) {
    Write-Host "[VLM Dual Server] legacy launcher not found: `"$LegacyScript`""
    exit 1
}

& $LegacyScript @args
exit $LASTEXITCODE
