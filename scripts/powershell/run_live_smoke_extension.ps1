param(
    [string]$IsaacRoot = "",
    [switch]$AutoRun,
    [string]$SmokeArgs = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$IsaacArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$ExtDir = Join-Path $RepoDir "exts"
$IsaacRootResolved = if ([string]::IsNullOrWhiteSpace($IsaacRoot)) {
    [System.IO.Path]::GetFullPath($(if ($env:ISAAC_SIM_ROOT) { $env:ISAAC_SIM_ROOT } else { "C:\isaac-sim" }))
} else {
    [System.IO.Path]::GetFullPath($IsaacRoot)
}
$IsaacFullApp = Join-Path $IsaacRootResolved "isaac-sim.bat"

if (-not (Test-Path -LiteralPath $IsaacFullApp)) {
    throw "[Live Smoke Extension] Isaac Full App launcher not found: $IsaacFullApp"
}
if (-not (Test-Path -LiteralPath $ExtDir)) {
    throw "[Live Smoke Extension] extension folder not found: $ExtDir"
}

$ArgumentList = @(
    "--ext-folder", $ExtDir,
    "--enable", "isaac.aura.live_smoke"
) + $IsaacArgs

if ($AutoRun) {
    $env:ISAAC_AURA_LIVE_SMOKE_AUTO_RUN = "1"
    $env:ISAAC_AURA_LIVE_SMOKE_ARGS = $SmokeArgs
}

try {
    Write-Host "[Live Smoke Extension] isaac_root=$IsaacRootResolved"
    Write-Host "[Live Smoke Extension] ext_folder=$ExtDir"
    Write-Host "[Live Smoke Extension] enable=isaac.aura.live_smoke"
    if ($AutoRun) {
        Write-Host "[Live Smoke Extension] auto-run enabled"
    } else {
        Write-Host "[Live Smoke Extension] open the Full App, then use menu: Isaac Aura > Run Live Smoke"
    }
    & $IsaacFullApp @ArgumentList
    exit $LASTEXITCODE
}
finally {
    if ($AutoRun) {
        Remove-Item Env:ISAAC_AURA_LIVE_SMOKE_AUTO_RUN -ErrorAction SilentlyContinue
        Remove-Item Env:ISAAC_AURA_LIVE_SMOKE_ARGS -ErrorAction SilentlyContinue
    }
}
