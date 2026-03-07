$ErrorActionPreference = "Stop"

# Example:
#   .\scripts\powershell\run_memory_agent.ps1 --bus zmq --control-endpoint tcp://127.0.0.1:5560 --telemetry-endpoint tcp://127.0.0.1:5561 --frame-source auto

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$EntryModule = "apps.memory_agent_app"
$PathSep = [System.IO.Path]::PathSeparator

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $PythonExe -m $EntryModule @args
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
