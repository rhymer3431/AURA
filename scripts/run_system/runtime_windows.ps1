param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 18096,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = "$repoRoot\src;$($env:PYTHONPATH)"

& $Python -m runtime.api.serve_runtime `
    --host $BindHost `
    --port $Port `
    --repo-root $repoRoot
