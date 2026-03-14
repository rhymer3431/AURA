param(
    [string]$EnvName = "aura-system2-lora",
    [string]$Config = "experiments/system2_memory_lora/system2_memory_lora.yaml",
    [string]$ModelNameOrPath = "",
    [string]$DatasetDir = "",
    [string]$OutputDir = "",
    [switch]$ValidateOnly
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
    $Args = @("-m", "inference.training.system2_memory_lora_train", "--config", $Config)
    if (-not [string]::IsNullOrWhiteSpace($ModelNameOrPath)) { $Args += @("--model-name-or-path", $ModelNameOrPath) }
    if (-not [string]::IsNullOrWhiteSpace($DatasetDir)) { $Args += @("--dataset-dir", $DatasetDir) }
    if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { $Args += @("--output-dir", $OutputDir) }
    if ($ValidateOnly) { $Args += "--validate-only" }
    conda run -n $EnvName python @Args
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
