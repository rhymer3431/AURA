param(
    [string]$EnvName = "aura-system2-lora",
    [string]$EnvironmentFile = "experiments/system2_memory_lora/environment.windows.yml",
    [switch]$SkipEditableInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))

$CondaExe = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $CondaExe) {
    throw "conda was not found in PATH. Open an Anaconda PowerShell prompt or add conda to PATH."
}

Push-Location $RepoDir
try {
    $EnvExists = [bool](conda env list | Select-String -Pattern ("^" + [regex]::Escape($EnvName) + "\\s"))
    if ($EnvExists) {
        conda env update --name $EnvName --file $EnvironmentFile --prune
    } else {
        conda env create --name $EnvName --file $EnvironmentFile
    }
    if (-not $SkipEditableInstall) {
        conda run -n $EnvName python -m pip install -e .
    }
    conda run -n $EnvName python -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()})"
}
finally {
    Pop-Location
}
