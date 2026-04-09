param(
    [string]$Model = "",

    [int]$Port = 8093,

    [int]$GpuLayers = 999,

    [int]$CtxSize = 1024,

    [string]$CacheTypeK = "q8_0",

    [string]$CacheTypeV = "q8_0"
)

$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $repoRoot "artifacts\models\Qwen3-1.7B-Q4_K_M-Instruct.gguf"
}

$llamaHome = if ($env:LLAMA_CPP_HOME) {
    $env:LLAMA_CPP_HOME
} else {
    Join-Path $repoRoot "llama.cpp"
}

$llamaServer = Join-Path $llamaHome "llama-server.exe"

if (-not (Test-Path $llamaServer)) {
    throw "llama-server.exe not found: $llamaServer"
}

if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

$args = @(
    "-m", $Model,
    "--jinja",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--reasoning-format", "none",
    "-ngl", $GpuLayers,
    "-c", $CtxSize,
    "-ctk", $CacheTypeK,
    "-ctv", $CacheTypeV,
    "--host", "127.0.0.1",
    "--port", $Port
)

& $llamaServer @args
