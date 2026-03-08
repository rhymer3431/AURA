param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))

function Convert-ToPythonStringLiteral {
    param([string]$Value)
    $Escaped = $Value.Replace("\", "\\").Replace("'", "\\'")
    return "'" + $Escaped + "'"
}

if ($ViewerArgs.Count -eq 0) {
    $ViewerArgs = @("--depth-max-m", "5.0")
}

$PythonArgs = @()
foreach ($Arg in $ViewerArgs) {
    $PythonArgs += "    $(Convert-ToPythonStringLiteral $Arg),"
}
$PythonArgsLiteral = [string]::Join("`n", $PythonArgs)

Push-Location $RepoDir
try {
    Write-Host "[Depth View Attach] External process attach is not supported."
    Write-Host "[Depth View Attach] Run the following inside Isaac Sim Script Editor:"
    $Snippet = @"
from apps.editor_depth_viewer_entry import run_depth_viewer
run_depth_viewer([
$PythonArgsLiteral
])
"@
    Write-Host $Snippet
    exit 1
}
finally {
    Pop-Location
}
