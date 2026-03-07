param(
    [string]$IsaacRoot = "",
    [string]$DiagnosticsPath = ".\tmp\process_logs\live_smoke\attach_diagnostics.json",
    [string]$ArtifactsDir = ".\tmp\process_logs\live_smoke",
    [switch]$ExtensionMode,
    [switch]$UseLegacyAttachAlias,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SmokeArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$IsaacRootResolved = if ([string]::IsNullOrWhiteSpace($IsaacRoot)) {
    [System.IO.Path]::GetFullPath($(if ($env:ISAAC_SIM_ROOT) { $env:ISAAC_SIM_ROOT } else { "C:\isaac-sim" }))
} else {
    [System.IO.Path]::GetFullPath($IsaacRoot)
}
$LaunchMode = if ($ExtensionMode) { "extension_mode" } elseif ($UseLegacyAttachAlias) { "full_app_attach" } else { "editor_assisted" }
$ExtensionLauncher = Join-Path $ScriptDir "run_live_smoke_extension.ps1"

Push-Location $RepoDir
try {
    Write-Host "[Live Smoke Attach] launch_mode=$LaunchMode"
    if ($UseLegacyAttachAlias) {
        Write-Host "[Live Smoke Attach] full_app_attach is deprecated. editor_assisted is the official in-editor smoke mode."
    }
    Write-Host "[Live Smoke Attach] External process attach is not supported."
    Write-Host "[Live Smoke Attach] Official in-editor paths are:"
    Write-Host "[Live Smoke Attach]   1) Script Editor / in-editor Python -> apps.editor_smoke_entry.run_editor_smoke(...)"
    Write-Host "[Live Smoke Attach]   2) Extension path -> scripts/powershell/run_live_smoke_extension.ps1"
    if ($ExtensionMode) {
        if (-not (Test-Path -LiteralPath $ExtensionLauncher)) {
            throw "[Live Smoke Attach] extension launcher not found: $ExtensionLauncher"
        }
        $JoinedSmokeArgs = [string]::Join(" ", $SmokeArgs)
        Write-Host "[Live Smoke Attach] forwarding to extension auto-run path"
        & $ExtensionLauncher -IsaacRoot $IsaacRootResolved -AutoRun -SmokeArgs $JoinedSmokeArgs
        exit $LASTEXITCODE
    }
    $Snippet = @"
from apps.editor_smoke_entry import run_editor_smoke
run_editor_smoke([
    "--diagnostics-path", r"$DiagnosticsPath",
    "--artifacts-dir", r"$ArtifactsDir",
])
"@
    Write-Host "[Live Smoke Attach] Run the following inside Isaac Sim Script Editor:"
    Write-Host $Snippet
    exit 1
}
finally {
    Pop-Location
}
