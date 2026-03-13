$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "apps.dual_server_app"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$CondaEnv = "fa2-cu130-py312"

$ListenHost = if ($env:DUAL_SERVER_HOST) { $env:DUAL_SERVER_HOST } else { "127.0.0.1" }
$Port = if ($env:DUAL_SERVER_PORT) { $env:DUAL_SERVER_PORT } else { "8890" }
$NavDPUrl = if ($env:DUAL_NAVDP_URL) { $env:DUAL_NAVDP_URL } else { "http://127.0.0.1:8888" }
$VLMUrl = if ($env:DUAL_VLM_URL) { $env:DUAL_VLM_URL } else { "http://127.0.0.1:8080" }
$VLMModel = if ($env:DUAL_VLM_MODEL) { $env:DUAL_VLM_MODEL } else { "InternVLA-N1-System2.Q4_K_M.gguf" }
$VLMTemperature = if ($env:DUAL_VLM_TEMPERATURE) { $env:DUAL_VLM_TEMPERATURE } else { "0.2" }
$VLMTopK = if ($env:DUAL_VLM_TOP_K) { $env:DUAL_VLM_TOP_K } else { "40" }
$VLMTopP = if ($env:DUAL_VLM_TOP_P) { $env:DUAL_VLM_TOP_P } else { "0.95" }
$VLMMinP = if ($env:DUAL_VLM_MIN_P) { $env:DUAL_VLM_MIN_P } else { "0.05" }
$VLMRepeatPenalty = if ($env:DUAL_VLM_REPEAT_PENALTY) { $env:DUAL_VLM_REPEAT_PENALTY } else { "1.1" }
$VLMNumHistory = if ($env:DUAL_VLM_NUM_HISTORY) { $env:DUAL_VLM_NUM_HISTORY } else { "8" }
$VLMMaxImagesPerRequest = if ($env:DUAL_VLM_MAX_IMAGES_PER_REQUEST) { $env:DUAL_VLM_MAX_IMAGES_PER_REQUEST } else { "3" }
$S2Mode = if ($env:DUAL_S2_MODE) { $env:DUAL_S2_MODE } else { "auto" }
$VLMTimeoutSec = if ($env:DUAL_VLM_TIMEOUT_SEC) { $env:DUAL_VLM_TIMEOUT_SEC } else { "35" }
$S2BackoffMaxSec = if ($env:DUAL_S2_BACKOFF_MAX_SEC) { $env:DUAL_S2_BACKOFF_MAX_SEC } else { "30" }
$CondaExe = if ($env:DUAL_CONDA_EXE) { $env:DUAL_CONDA_EXE } else { $DefaultCondaExe }

if (-not (Test-Path -LiteralPath $SrcDir)) {
    Write-Host "[Dual Server] src directory not found: `"$SrcDir`""
    exit 1
}

if (-not (Test-Path -LiteralPath $CondaExe)) {
    Write-Host "[Dual Server] conda executable not found: `"$CondaExe`""
    Write-Host "[Dual Server] Set DUAL_CONDA_EXE to your conda.exe path."
    exit 1
}

Write-Host "[Dual Server] Starting dual orchestrator"
Write-Host "[Dual Server] conda-exe=`"$CondaExe`""
Write-Host "[Dual Server] env=`"$CondaEnv`""
Write-Host "[Dual Server] module=`"$EntryModule`""
Write-Host "[Dual Server] host=$ListenHost port=$Port"
Write-Host "[Dual Server] navdp-url=$NavDPUrl vlm-url=$VLMUrl vlm-model=$VLMModel s2-mode=$S2Mode"

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $CondaExe run --no-capture-output -n $CondaEnv python -m $EntryModule --host $ListenHost --port $Port --navdp-url $NavDPUrl --vlm-url $VLMUrl --vlm-model $VLMModel --vlm-temperature $VLMTemperature --vlm-top-k $VLMTopK --vlm-top-p $VLMTopP --vlm-min-p $VLMMinP --vlm-repeat-penalty $VLMRepeatPenalty --vlm-num-history $VLMNumHistory --vlm-max-images-per-request $VLMMaxImagesPerRequest --s2-mode $S2Mode --vlm-timeout-sec $VLMTimeoutSec --s2-failure-backoff-max-sec $S2BackoffMaxSec @args
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
