param(
    [Alias("Host")]
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8080,
    [string]$ModelPath = "artifacts/models/InternVLA-N1-System2.Q4_K_M.gguf",
    [string]$MmprojPath = "artifacts/models/InternVLA-N1-System2.mmproj-Q8_0.gguf",
    [string]$ChatTemplateFile = "scripts/powershell/internvla_system2_chat_template.jinja",
    [string]$LlamaServer = "llama-server",
    [string]$HFRepo = "mradermacher/InternVLA-N1-System2-GGUF",
    [string]$HFMMProjFile = "InternVLA-N1-System2.mmproj-Q8_0.gguf",
    [int]$ContextSize = 8192,
    [string]$GpuLayers = "auto",
    [int]$ReasoningBudget = 0,
    [switch]$SkipMmprojDownload,
    [switch]$ForceCudaRuntimeRefresh
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))

if (-not $PSBoundParameters.ContainsKey("ListenHost") -and $env:INTERNVLA_HOST) { $ListenHost = $env:INTERNVLA_HOST }
if (-not $PSBoundParameters.ContainsKey("Port") -and $env:INTERNVLA_PORT) { $Port = [int]$env:INTERNVLA_PORT }
if (-not $PSBoundParameters.ContainsKey("ModelPath") -and $env:INTERNVLA_MODEL_PATH) { $ModelPath = $env:INTERNVLA_MODEL_PATH }
if (-not $PSBoundParameters.ContainsKey("MmprojPath") -and $env:INTERNVLA_MMPROJ_PATH) { $MmprojPath = $env:INTERNVLA_MMPROJ_PATH }
if (-not $PSBoundParameters.ContainsKey("ChatTemplateFile") -and $env:INTERNVLA_CHAT_TEMPLATE_FILE) { $ChatTemplateFile = $env:INTERNVLA_CHAT_TEMPLATE_FILE }
if (-not $PSBoundParameters.ContainsKey("LlamaServer") -and $env:LLAMA_SERVER_EXE) { $LlamaServer = $env:LLAMA_SERVER_EXE }
if (-not $PSBoundParameters.ContainsKey("HFRepo") -and $env:INTERNVLA_HF_REPO) { $HFRepo = $env:INTERNVLA_HF_REPO }
if (-not $PSBoundParameters.ContainsKey("HFMMProjFile") -and $env:INTERNVLA_HF_MMPROJ_FILE) { $HFMMProjFile = $env:INTERNVLA_HF_MMPROJ_FILE }
if (-not $PSBoundParameters.ContainsKey("ContextSize") -and $env:INTERNVLA_CTX_SIZE) { $ContextSize = [int]$env:INTERNVLA_CTX_SIZE }
if (-not $PSBoundParameters.ContainsKey("GpuLayers") -and $env:INTERNVLA_GPU_LAYERS) { $GpuLayers = $env:INTERNVLA_GPU_LAYERS }
if (-not $PSBoundParameters.ContainsKey("ReasoningBudget") -and $env:INTERNVLA_REASONING_BUDGET) { $ReasoningBudget = [int]$env:INTERNVLA_REASONING_BUDGET }

function Resolve-ProjectPath([string]$InputPath) {
    if ([string]::IsNullOrWhiteSpace($InputPath)) { throw "Path cannot be empty." }
    if ([System.IO.Path]::IsPathRooted($InputPath)) { return [System.IO.Path]::GetFullPath($InputPath) }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoDir $InputPath))
}

function Resolve-LlamaServerPath([string]$CommandOrPath) {
    if ([string]::IsNullOrWhiteSpace($CommandOrPath)) { return $null }

    $looksLikePath = $CommandOrPath.Contains("\") -or $CommandOrPath.Contains("/") -or $CommandOrPath.EndsWith(".exe")
    if ($looksLikePath) {
        $candidate = if ([System.IO.Path]::IsPathRooted($CommandOrPath)) {
            [System.IO.Path]::GetFullPath($CommandOrPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $RepoDir $CommandOrPath))
        }
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }

    $cmd = Get-Command $CommandOrPath -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    $fallbacks = @(
        (Join-Path $RepoDir "llama.cpp\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\Release\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\RelWithDebInfo\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\llama-server.exe"),
        (Join-Path $RepoDir "llama-server.exe")
    )
    foreach ($fallback in $fallbacks) {
        if (Test-Path -LiteralPath $fallback) { return [System.IO.Path]::GetFullPath($fallback) }
    }
    return $null
}

function Test-LlamaGpuBackend([string]$LlamaServerPath) {
    $serverDir = Split-Path -Parent $LlamaServerPath
    $gpuBackendFiles = @(
        "ggml-cuda.dll",
        "ggml-vulkan.dll",
        "ggml-hip.dll",
        "ggml-sycl.dll",
        "ggml-opencl.dll",
        "ggml-metal.metal"
    )
    foreach ($backendFile in $gpuBackendFiles) {
        if (Test-Path -LiteralPath (Join-Path $serverDir $backendFile)) { return $true }
    }
    return $false
}

function Add-ProcessPathEntry([string]$PathEntry) {
    if ([string]::IsNullOrWhiteSpace($PathEntry) -or -not (Test-Path -LiteralPath $PathEntry)) { return }
    $pathEntries = @($env:PATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($pathEntries -contains $PathEntry) { return }
    $env:PATH = "$PathEntry;$env:PATH"
}

function Get-CudaRuntimeZipPath([string]$BaseDir) {
    $candidates = @(
        (Join-Path $BaseDir "cudart-llama-bin-win-cuda-13.1-x64.zip"),
        (Join-Path $BaseDir "cudart-llama-bin-win-cuda-13.1-x64.zip.tmp")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return Join-Path $BaseDir "cudart-llama-bin-win-cuda-13.1-x64.zip"
}

function Ensure-Cuda131Runtime([string]$BaseDir, [string]$ServerDir, [switch]$ForceRefresh) {
    $requiredDlls = @("cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll")
    $missingDlls = @(
        $requiredDlls | Where-Object {
            $dllPath = Join-Path $ServerDir $_
            -not (Test-Path -LiteralPath $dllPath)
        }
    )

    if (-not $ForceRefresh -and $missingDlls.Count -eq 0) { return }

    $runtimeZip = Get-CudaRuntimeZipPath $BaseDir
    $runtimeUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b8191/cudart-llama-bin-win-cuda-13.1-x64.zip"

    if ($ForceRefresh -or -not (Test-Path -LiteralPath $runtimeZip)) {
        Write-Host "[InternVLA System2] downloading CUDA 13.1 runtime bundle"
        Write-Host "[InternVLA System2] url=$runtimeUrl"
        Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip
    }

    $extractDir = Join-Path $BaseDir "_cudart_extract"
    Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue

    try {
        Expand-Archive -LiteralPath $runtimeZip -DestinationPath $extractDir -Force
    } catch {
        Write-Host "[InternVLA System2] CUDA runtime bundle is missing or invalid: `"$runtimeZip`""
        Write-Host "[InternVLA System2] downloading a fresh runtime bundle and retrying"
        Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip
        Expand-Archive -LiteralPath $runtimeZip -DestinationPath $extractDir -Force
    }

    foreach ($dllName in $requiredDlls) {
        $sourcePath = Join-Path $extractDir $dllName
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Required CUDA runtime DLL missing from bundle: $dllName"
        }
        Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $ServerDir $dllName) -Force
    }
}

$ResolvedModelPath = Resolve-ProjectPath $ModelPath
$ResolvedMmprojPath = Resolve-ProjectPath $MmprojPath
$ResolvedChatTemplateFile = $null
if (-not [string]::IsNullOrWhiteSpace($ChatTemplateFile)) {
    $ResolvedChatTemplateFile = Resolve-ProjectPath $ChatTemplateFile
}
if (-not (Test-Path -LiteralPath $ResolvedModelPath)) {
    $fallbackModel = Resolve-ProjectPath "InternVLA-N1-System2.Q4_K_M.gguf"
    if (Test-Path -LiteralPath $fallbackModel) { $ResolvedModelPath = $fallbackModel }
}
if (-not (Test-Path -LiteralPath $ResolvedMmprojPath)) {
    $fallbackMmproj = Resolve-ProjectPath "InternVLA-N1-System2.mmproj-Q8_0.gguf"
    if (Test-Path -LiteralPath $fallbackMmproj) { $ResolvedMmprojPath = $fallbackMmproj }
}
$ResolvedLlamaServerPath = Resolve-LlamaServerPath $LlamaServer

if (-not (Test-Path -LiteralPath $ResolvedModelPath)) {
    Write-Host "[InternVLA System2] model not found: `"$ResolvedModelPath`""
    Write-Host "[InternVLA System2] Set INTERNVLA_MODEL_PATH or pass -ModelPath."
    exit 1
}

if (-not (Test-Path -LiteralPath $ResolvedMmprojPath)) {
    if ($SkipMmprojDownload) {
        Write-Host "[InternVLA System2] mmproj not found: `"$ResolvedMmprojPath`""
        Write-Host "[InternVLA System2] Remove -SkipMmprojDownload to auto-download it."
        exit 1
    }

    $mmprojDir = Split-Path -Parent $ResolvedMmprojPath
    if (-not [string]::IsNullOrWhiteSpace($mmprojDir)) { New-Item -ItemType Directory -Path $mmprojDir -Force | Out-Null }
    $downloadUrl = "https://huggingface.co/$($HFRepo)/resolve/main/$($HFMMProjFile)?download=true"
    Write-Host "[InternVLA System2] downloading mmproj from: $downloadUrl"
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $ResolvedMmprojPath
    } catch {
        Write-Host "[InternVLA System2] mmproj download failed: $($_.Exception.Message)"
        exit 1
    }
}

if (-not [string]::IsNullOrWhiteSpace($ResolvedChatTemplateFile) -and -not (Test-Path -LiteralPath $ResolvedChatTemplateFile)) {
    Write-Host "[InternVLA System2] chat template not found: `"$ResolvedChatTemplateFile`""
    Write-Host "[InternVLA System2] continuing without --chat-template-file"
    $ResolvedChatTemplateFile = $null
}

if ($null -eq $ResolvedLlamaServerPath -or -not (Test-Path -LiteralPath $ResolvedLlamaServerPath)) {
    Write-Host "[InternVLA System2] llama-server not found. Pass -LlamaServer or set LLAMA_SERVER_EXE."
    exit 1
}

$ServerDir = Split-Path -Parent $ResolvedLlamaServerPath
Ensure-Cuda131Runtime -BaseDir $RepoDir -ServerDir $ServerDir -ForceRefresh:$ForceCudaRuntimeRefresh
Add-ProcessPathEntry $ServerDir

if (-not (Test-LlamaGpuBackend $ResolvedLlamaServerPath)) {
    Write-Host "[InternVLA System2] llama-server GPU backend not found near: `"$ResolvedLlamaServerPath`""
    exit 1
}

$GpuLayerArg = if ($GpuLayers -eq "auto") { "-1" } else { $GpuLayers }

Write-Host "[InternVLA System2] Starting llama-server"
Write-Host "[InternVLA System2] llama-server=`"$ResolvedLlamaServerPath`""
Write-Host "[InternVLA System2] model=`"$ResolvedModelPath`""
Write-Host "[InternVLA System2] mmproj=`"$ResolvedMmprojPath`""
if ([string]::IsNullOrWhiteSpace($ResolvedChatTemplateFile)) {
    Write-Host "[InternVLA System2] chat-template=disabled"
} else {
    Write-Host "[InternVLA System2] chat-template=`"$ResolvedChatTemplateFile`""
}
Write-Host "[InternVLA System2] host=$ListenHost port=$Port"
Write-Host "[InternVLA System2] reasoning-budget=$ReasoningBudget"

Push-Location $RepoDir
try {
    $ServerArgs = @(
        "--host", $ListenHost,
        "--port", $Port,
        "--model", $ResolvedModelPath,
        "--mmproj", $ResolvedMmprojPath,
        "--ctx-size", $ContextSize,
        "--gpu-layers", $GpuLayerArg,
        "--reasoning-budget", $ReasoningBudget
    )
    if (-not [string]::IsNullOrWhiteSpace($ResolvedChatTemplateFile)) {
        $ServerArgs += @("--chat-template-file", $ResolvedChatTemplateFile)
    }
    $ServerArgs += $args

    & $ResolvedLlamaServerPath @ServerArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
