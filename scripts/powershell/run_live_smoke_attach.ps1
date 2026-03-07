param(
    [string]$IsaacRoot = "",
    [string]$DiagnosticsPath = ".\tmp\process_logs\live_smoke\attach_diagnostics.json",
    [string]$ArtifactsDir = ".\tmp\process_logs\live_smoke",
    [switch]$ExtensionMode,
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
$LaunchMode = if ($ExtensionMode) { "extension_mode" } else { "full_app_attach" }

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    throw "[Live Smoke Attach] Isaac python launcher not found: $IsaacPython"
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    Write-Host "[Live Smoke Attach] launch_mode=$LaunchMode"
    Write-Host "[Live Smoke Attach] This path expects a running Isaac Sim Full App / Kit stage."
    & $IsaacPython -m apps.live_smoke_app --mode smoke --launch-mode $LaunchMode --diagnostics-path $DiagnosticsPath --artifacts-dir $ArtifactsDir @SmokeArgs
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
