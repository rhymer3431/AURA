$ErrorActionPreference = "Stop"

# Example:
#   .\scripts\powershell\run_local_stack.ps1 --command "따라와" --scene person --frame-source auto

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$EntryModule = "apps.local_stack_app"
$PathSep = [System.IO.Path]::PathSeparator

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    Write-Warning "run_local_stack.ps1 is deprecated and pending removal. Prefer run_aura_runtime.ps1 or run_memory_agent.ps1 --loopback."
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
