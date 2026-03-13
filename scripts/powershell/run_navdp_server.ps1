$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "apps.navdp_server_app"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultCkptCandidates = @(
    (Join-Path $RepoDir "artifacts\models\navdp-cross-modal.ckpt"),
    (Join-Path $RepoDir "navdp-weights.ckpt")
)
$ResolvedDefaultCheckpoint = $DefaultCkptCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $ResolvedDefaultCheckpoint) {
    $ResolvedDefaultCheckpoint = $DefaultCkptCandidates[0]
}
$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$CondaEnv = "fa2-cu130-py312"

$Port = if ($env:NAVDP_PORT) { $env:NAVDP_PORT } else { "8888" }
$Checkpoint = if ($env:NAVDP_CHECKPOINT) {
    $env:NAVDP_CHECKPOINT
} else {
    $ResolvedDefaultCheckpoint
}
$CondaExe = if ($env:NAVDP_CONDA_EXE) { $env:NAVDP_CONDA_EXE } else { $DefaultCondaExe }

if (-not (Test-Path -LiteralPath $SrcDir)) {
    Write-Host "[NavDP Server] src directory not found: `"$SrcDir`""
    exit 1
}

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    Write-Host "[NavDP Server] Checkpoint not found: `"$Checkpoint`""
    Write-Host "[NavDP Server] Set NAVDP_CHECKPOINT or pass --checkpoint explicitly."
    exit 1
}

if (-not (Test-Path -LiteralPath $CondaExe)) {
    Write-Host "[NavDP Server] conda executable not found: `"$CondaExe`""
    Write-Host "[NavDP Server] Set NAVDP_CONDA_EXE to your conda.exe path."
    exit 1
}

Write-Host "[NavDP Server] Starting NavDP server"
Write-Host "[NavDP Server] conda-exe=`"$CondaExe`""
Write-Host "[NavDP Server] env=`"$CondaEnv`""
Write-Host "[NavDP Server] module=`"$EntryModule`""
Write-Host "[NavDP Server] port=$Port"
Write-Host "[NavDP Server] checkpoint=`"$Checkpoint`""

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $CondaExe run --no-capture-output -n $CondaEnv python -m $EntryModule --port $Port --checkpoint $Checkpoint @args
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
