$ErrorActionPreference = "Stop"

# Example:
#   .\scripts\powershell\run_isaac_bridge.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --frame-source auto

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultIsaacPython = "C:\isaac-sim\python.bat"
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$EntryModule = "apps.isaac_bridge_app"

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    Write-Host "[Isaac Bridge] Isaac python launcher not found: `"$IsaacPython`""
    Write-Host "[Isaac Bridge] Falling back to `"$PythonExe`" for direct IPC loopback or bridge-only runs."
    $IsaacPython = $PythonExe
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $IsaacPython -m $EntryModule @args
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
