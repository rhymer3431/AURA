$ErrorActionPreference = "Stop"

# Example:
#   .\scripts\powershell\run_g1_viewer.ps1 --control-endpoint tcp://127.0.0.1:5580 --telemetry-endpoint tcp://127.0.0.1:5581 --shm-name g1_view_frames
#   $env:G1_VIEWER_PYTHON_EXE="C:\Users\mango\anaconda3\envs\viewer\python.exe"
#   .\scripts\powershell\run_g1_viewer.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$CondaEnv = "fa2-cu130-py312"
$PythonExe = if ($env:G1_VIEWER_PYTHON_EXE) { $env:G1_VIEWER_PYTHON_EXE } else { "" }
$CondaExe = if ($env:G1_VIEWER_CONDA_EXE) { $env:G1_VIEWER_CONDA_EXE } else { $DefaultCondaExe }
$EntryModule = "apps.g1_viewer_app"
$PathSep = [System.IO.Path]::PathSeparator

if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    Write-Host "[G1 Viewer] python=`"$PythonExe`""
} else {
    Write-Host "[G1 Viewer] conda-exe=`"$CondaExe`""
    Write-Host "[G1 Viewer] env=`"$CondaEnv`""
}

if (-not [string]::IsNullOrWhiteSpace($PythonExe) -and (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue))) {
    Write-Host "[G1 Viewer] python executable not found: `"$PythonExe`""
    Write-Host "[G1 Viewer] Set G1_VIEWER_PYTHON_EXE to a valid python path or unset it to use conda."
    exit 1
}

if ([string]::IsNullOrWhiteSpace($PythonExe) -and (-not (Test-Path -LiteralPath $CondaExe))) {
    Write-Host "[G1 Viewer] conda executable not found: `"$CondaExe`""
    Write-Host "[G1 Viewer] Set G1_VIEWER_CONDA_EXE to your conda.exe path."
    exit 1
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        & $PythonExe -m $EntryModule @args
    } else {
        & $CondaExe run --no-capture-output -n $CondaEnv python -m $EntryModule @args
    }
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
