$ErrorActionPreference = "Stop"

# Example:
#   .\scripts\powershell\run_g1_viewer.ps1 --control-endpoint tcp://127.0.0.1:5580 --telemetry-endpoint tcp://127.0.0.1:5581 --shm-name g1_view_frames
#   $env:G1_VIEWER_PYTHON_EXE="C:\Users\mango\anaconda3\envs\viewer\python.exe"
#   .\scripts\powershell\run_g1_viewer.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PythonExe = if ($env:G1_VIEWER_PYTHON_EXE) { $env:G1_VIEWER_PYTHON_EXE } elseif ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$EntryModule = "apps.g1_viewer_app"
$PathSep = [System.IO.Path]::PathSeparator

Write-Host "[G1 Viewer] python=`"$PythonExe`""

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
