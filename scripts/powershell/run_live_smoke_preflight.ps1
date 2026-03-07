param(
    [string]$IsaacRoot = "",
    [string]$DiagnosticsPath = ".\tmp\process_logs\live_smoke\preflight_diagnostics.json",
    [string]$ArtifactsDir = ".\tmp\process_logs\live_smoke",
    [switch]$ClearCache,
    [switch]$Warmup,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SmokeArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$IsaacRootResolved = if ([string]::IsNullOrWhiteSpace($IsaacRoot)) {
    [System.IO.Path]::GetFullPath($(if ($env:ISAAC_SIM_ROOT) { $env:ISAAC_SIM_ROOT } else { "C:\isaac-sim" }))
} else {
    [System.IO.Path]::GetFullPath($IsaacRoot)
}
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { Join-Path $IsaacRootResolved "python.bat" }
$ClearCacheScript = Join-Path $IsaacRootResolved "clear_caches.bat"
$WarmupScript = Join-Path $IsaacRootResolved "warmup.bat"

if (-not (Test-Path -LiteralPath $IsaacRootResolved)) {
    throw "[Live Smoke Preflight] Isaac root not found: $IsaacRootResolved"
}
if (-not (Test-Path -LiteralPath $IsaacPython)) {
    throw "[Live Smoke Preflight] Isaac python launcher not found: $IsaacPython"
}

if ($ClearCache) {
    if (-not (Test-Path -LiteralPath $ClearCacheScript)) {
        throw "[Live Smoke Preflight] clear_caches.bat not found: $ClearCacheScript"
    }
    Write-Host "[Live Smoke Preflight] clearing Isaac caches via $ClearCacheScript"
    & $ClearCacheScript
}
if ($Warmup) {
    if (-not (Test-Path -LiteralPath $WarmupScript)) {
        throw "[Live Smoke Preflight] warmup.bat not found: $WarmupScript"
    }
    Write-Host "[Live Smoke Preflight] warming up Isaac via $WarmupScript"
    & $WarmupScript
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    Write-Host "[Live Smoke Preflight] isaac_root=$IsaacRootResolved"
    Write-Host "[Live Smoke Preflight] isaac_python=$IsaacPython"
    & $IsaacPython -m apps.live_smoke_app --mode preflight --diagnostics-path $DiagnosticsPath --artifacts-dir $ArtifactsDir @SmokeArgs
    exit $LASTEXITCODE
}
finally {
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Pop-Location
}
