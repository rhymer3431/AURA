$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LegacyScript = Join-Path $ScriptDir "legacy\run_navdp_server.ps1"

if (-not (Test-Path -LiteralPath $LegacyScript)) {
    Write-Host "[NavDP Server] legacy launcher not found: `"$LegacyScript`""
    exit 1
}

& $LegacyScript @args
exit $LASTEXITCODE
