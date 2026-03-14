param(
    [string]$OutputDir = "artifacts/datasets/system2_memory_lora_seed",
    [int]$TrainCount = 24,
    [int]$ValCount = 8,
    [int]$TestCount = 8,
    [int]$ImageSize = 320,
    [int]$Seed = 7,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$PreviousPythonPath = $env:PYTHONPATH

Push-Location $RepoDir
try {
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        (Join-Path $RepoDir "src")
    } else {
        (Join-Path $RepoDir "src") + ";" + $PreviousPythonPath
    }
    & $PythonExe -m inference.training.system2_memory_lora build-seed `
        --output-dir $OutputDir `
        --train-count $TrainCount `
        --val-count $ValCount `
        --test-count $TestCount `
        --image-size $ImageSize `
        --seed $Seed

    & $PythonExe -m inference.training.system2_memory_lora validate --dataset-dir $OutputDir
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
