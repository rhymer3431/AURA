param(
    [ValidateSet("all", "nav", "s2", "dual", "runtime")]
    [string]$Component = "all",
    [int]$StartupTimeoutSec = 180,
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$ComponentArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SelfScriptPath = $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$ProcessLogDir = Join-Path $RepoDir "tmp\process_logs\system"
$PowerShellExe = Join-Path $PSHOME "powershell.exe"

$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$DefaultCondaEnv = "fa2-cu130-py312"
$DefaultIsaacPython = "C:\isaac-sim\python.bat"

$NavEntryModule = "apps.navdp_server_app"
$DualEntryModule = "apps.dual_server_app"
$RuntimeEntryModule = "runtime.aura_runtime"

$NavPort = if ($env:NAVDP_PORT) { [int]$env:NAVDP_PORT } else { 8888 }
$System2Host = if ($env:INTERNVLA_HOST) { $env:INTERNVLA_HOST } else { "127.0.0.1" }
$System2Port = if ($env:INTERNVLA_PORT) { [int]$env:INTERNVLA_PORT } else { 8080 }
$DualHost = if ($env:DUAL_SERVER_HOST) { $env:DUAL_SERVER_HOST } else { "127.0.0.1" }
$DualPort = if ($env:DUAL_SERVER_PORT) { [int]$env:DUAL_SERVER_PORT } else { 8890 }
$NavBaseUrl = if ($env:DUAL_NAVDP_URL) { $env:DUAL_NAVDP_URL } else { "http://127.0.0.1:$NavPort" }
$System2BaseUrl = if ($env:DUAL_VLM_URL) { $env:DUAL_VLM_URL } else { "http://$System2Host`:$System2Port" }
$DualBaseUrl = "http://$DualHost`:$DualPort"
$CondaExe = if ($env:AURA_CONDA_EXE) { $env:AURA_CONDA_EXE } else { $DefaultCondaExe }
$CondaEnv = if ($env:AURA_CONDA_ENV) { $env:AURA_CONDA_ENV } else { $DefaultCondaEnv }
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }

$ForwardArgs = @()
foreach ($Arg in @($ComponentArgs)) {
    if ($null -eq $Arg) {
        continue
    }
    $StringArg = [string]$Arg
    if ([string]::IsNullOrWhiteSpace($StringArg)) {
        continue
    }
    $ForwardArgs += $StringArg
}

function Test-LaunchArgPresent {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    foreach ($LaunchArg in $InputArgs) {
        foreach ($Name in $Names) {
            if ($LaunchArg -eq $Name -or $LaunchArg.StartsWith("$Name=")) {
                return $true
            }
        }
    }
    return $false
}

function Get-LaunchArgValue {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$DefaultValue
    )

    $Resolved = $DefaultValue
    for ($Index = 0; $Index -lt $InputArgs.Length; $Index += 1) {
        $LaunchArg = $InputArgs[$Index]
        foreach ($Name in $Names) {
            if ($LaunchArg -eq $Name) {
                if (($Index + 1) -lt $InputArgs.Length) {
                    $Resolved = [string]$InputArgs[$Index + 1]
                }
                continue
            }
            if ($LaunchArg.StartsWith("$Name=")) {
                $Resolved = $LaunchArg.Substring($Name.Length + 1)
            }
        }
    }
    return $Resolved
}

function Remove-LaunchArgs {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    $Filtered = New-Object System.Collections.Generic.List[string]
    for ($Index = 0; $Index -lt $InputArgs.Length; $Index += 1) {
        $LaunchArg = $InputArgs[$Index]
        $Matched = $false
        foreach ($Name in $Names) {
            if ($LaunchArg -eq $Name) {
                $Matched = $true
                if (($Index + 1) -lt $InputArgs.Length) {
                    $Index += 1
                }
                break
            }
            if ($LaunchArg.StartsWith("$Name=")) {
                $Matched = $true
                break
            }
        }
        if (-not $Matched) {
            $Filtered.Add($LaunchArg) | Out-Null
        }
    }

    return @($Filtered)
}

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputPath
    )

    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        throw "Path cannot be empty."
    }
    if ([System.IO.Path]::IsPathRooted($InputPath)) {
        return [System.IO.Path]::GetFullPath($InputPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoDir $InputPath))
}

function Invoke-CondaModule {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EntryModule,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    if (-not (Test-Path -LiteralPath $SrcDir)) {
        throw "src directory not found: $SrcDir"
    }
    if (-not (Test-Path -LiteralPath $CondaExe)) {
        throw "conda executable not found: $CondaExe"
    }

    Push-Location $SrcDir
    $PreviousPythonPath = $env:PYTHONPATH
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        $env:PYTHONPATH = $SrcDir
    }
    else {
        $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
    }
    try {
        & $CondaExe run --no-capture-output -n $CondaEnv python -m $EntryModule @Arguments
        return $LASTEXITCODE
    }
    finally {
        if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHONPATH = $PreviousPythonPath
        }
        Pop-Location
    }
}

function Wait-TcpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Host,
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [int]$TimeoutSec = 60,
        [System.Diagnostics.Process]$Process = $null
    )

    $Deadline = (Get-Date).AddSeconds([Math]::Max($TimeoutSec, 1))
    do {
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                throw "$Name exited before becoming ready (exit code $($Process.ExitCode))."
            }
        }

        try {
            $Client = [System.Net.Sockets.TcpClient]::new()
            $Async = $Client.BeginConnect($Host, $Port, $null, $null)
            if ($Async.AsyncWaitHandle.WaitOne(1000)) {
                $Client.EndConnect($Async)
                $Client.Close()
                return
            }
            $Client.Close()
        }
        catch {
        }

        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $Deadline)

    throw "$Name did not become ready at ${Host}:${Port} within ${TimeoutSec}s."
}

function Stop-ManagedProcess {
    param(
        [System.Diagnostics.Process]$Process
    )

    if ($null -eq $Process) {
        return
    }
    try {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
    }
}

function Start-BackgroundSelf {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$TargetComponent,
        [string[]]$Arguments = @()
    )

    New-Item -ItemType Directory -Force -Path $ProcessLogDir | Out-Null
    $SafeName = ($Name -replace '[^A-Za-z0-9_-]', '_')
    $StdoutLog = Join-Path $ProcessLogDir "${SafeName}.stdout.log"
    $StderrLog = Join-Path $ProcessLogDir "${SafeName}.stderr.log"
    $ArgList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $SelfScriptPath,
        "-Component", $TargetComponent
    ) + $Arguments
    $Process = Start-Process -FilePath $PowerShellExe -ArgumentList $ArgList -WorkingDirectory $RepoDir -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
    Write-Host "[AURA_SYSTEM] started $Name pid=$($Process.Id)"
    Write-Host "[AURA_SYSTEM] logs: $StdoutLog / $StderrLog"
    return $Process
}

function Resolve-LlamaServerPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandOrPath
    )

    $LooksLikePath = $CommandOrPath.Contains("\") -or $CommandOrPath.Contains("/") -or $CommandOrPath.EndsWith(".exe")
    if ($LooksLikePath) {
        $Candidate = if ([System.IO.Path]::IsPathRooted($CommandOrPath)) {
            [System.IO.Path]::GetFullPath($CommandOrPath)
        }
        else {
            [System.IO.Path]::GetFullPath((Join-Path $RepoDir $CommandOrPath))
        }
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }

    $Command = Get-Command $CommandOrPath -ErrorAction SilentlyContinue
    if ($Command -and $Command.Source) {
        return $Command.Source
    }

    $Fallbacks = @(
        (Join-Path $RepoDir "llama.cpp\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\Release\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\RelWithDebInfo\llama-server.exe"),
        (Join-Path $RepoDir "llama.cpp\build\bin\llama-server.exe"),
        (Join-Path $RepoDir "llama-server.exe")
    )
    foreach ($Fallback in $Fallbacks) {
        if (Test-Path -LiteralPath $Fallback) {
            return [System.IO.Path]::GetFullPath($Fallback)
        }
    }
    return $null
}

function Test-LlamaGpuBackend {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LlamaServerPath
    )

    $ServerDir = Split-Path -Parent $LlamaServerPath
    foreach ($BackendFile in @("ggml-cuda.dll", "ggml-vulkan.dll", "ggml-hip.dll", "ggml-sycl.dll", "ggml-opencl.dll", "ggml-metal.metal")) {
        if (Test-Path -LiteralPath (Join-Path $ServerDir $BackendFile)) {
            return $true
        }
    }
    return $false
}

function Add-ProcessPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathEntry
    )

    if ([string]::IsNullOrWhiteSpace($PathEntry) -or -not (Test-Path -LiteralPath $PathEntry)) {
        return
    }
    $PathEntries = @($env:PATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($PathEntries -contains $PathEntry) {
        return
    }
    $env:PATH = "$PathEntry;$env:PATH"
}

function Resolve-SceneSelection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedPreset
    )

    $WarehouseEnvUrl = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    $InteriorAgentSceneUsd = Join-Path $RepoDir "datasets\InteriorAgent\kujiale_0004\kujiale_0004_navila_sanitized.usda"
    $InteriorAgentKujiale3SceneUsd = Join-Path $RepoDir "datasets\InteriorAgent\kujiale_0003\kujiale_0003.usda"
    $Normalized = $SelectedPreset.Trim().ToLowerInvariant()

    switch ($Normalized) {
        "" {
            return @{ SceneUsd = ""; EnvUrl = $WarehouseEnvUrl }
        }
        "warehouse" {
            return @{ SceneUsd = ""; EnvUrl = $WarehouseEnvUrl }
        }
        "interioragent" {
            return @{ SceneUsd = $InteriorAgentSceneUsd; EnvUrl = "" }
        }
        "interior" {
            return @{ SceneUsd = $InteriorAgentSceneUsd; EnvUrl = "" }
        }
        "interior agent kujiale 3" {
            return @{ SceneUsd = $InteriorAgentKujiale3SceneUsd; EnvUrl = "" }
        }
        "interioragent kujiale 3" {
            return @{ SceneUsd = $InteriorAgentKujiale3SceneUsd; EnvUrl = "" }
        }
        "interioragent_kujiale3" {
            return @{ SceneUsd = $InteriorAgentKujiale3SceneUsd; EnvUrl = "" }
        }
        "interioragent-kujiale3" {
            return @{ SceneUsd = $InteriorAgentKujiale3SceneUsd; EnvUrl = "" }
        }
        default {
            throw "unsupported scene preset: $SelectedPreset"
        }
    }
}

function Build-RuntimeLaunchArgs {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$InputArgs
    )

    $DefaultPolicyCandidates = @(
        (Join-Path $RepoDir "artifacts\models\g1_policy_fp16.engine"),
        (Join-Path $RepoDir "artifacts\models\g1_policy_fp32.engine"),
        (Join-Path $RepoDir "artifacts\models\policy.onnx"),
        (Join-Path $RepoDir "policy.onnx"),
        (Join-Path $RepoDir "src\locomotion\models\policy_fp16.engine")
    )
    $PolicyPath = if ($env:G1_POINTGOAL_POLICY) { $env:G1_POINTGOAL_POLICY } else { ($DefaultPolicyCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1) }
    if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
        $PolicyPath = $DefaultPolicyCandidates[0]
    }
    $RobotUsd = if ($env:G1_POINTGOAL_ROBOT_USD) { $env:G1_POINTGOAL_ROBOT_USD } else { (Join-Path $RepoDir "src\locomotion\g1\g1_d455.usd") }
    $ScenePreset = if ($env:G1_POINTGOAL_SCENE_PRESET) { $env:G1_POINTGOAL_SCENE_PRESET } else { "warehouse" }
    $PlannerMode = if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "interactive" }
    $GoalX = if ($env:G1_POINTGOAL_GOAL_X) { $env:G1_POINTGOAL_GOAL_X } else { "2.0" }
    $GoalY = if ($env:G1_POINTGOAL_GOAL_Y) { $env:G1_POINTGOAL_GOAL_Y } else { "0.0" }
    $LaunchMode = if ($env:G1_POINTGOAL_LAUNCH_MODE) { $env:G1_POINTGOAL_LAUNCH_MODE } else { "" }
    $ServerUrl = if ($env:G1_POINTGOAL_SERVER_URL) { $env:G1_POINTGOAL_SERVER_URL } else { $NavBaseUrl }
    $DualServerUrl = if ($env:G1_POINTGOAL_DUAL_SERVER_URL) { $env:G1_POINTGOAL_DUAL_SERVER_URL } else { $DualBaseUrl }
    $ViewerControlEndpoint = if ($env:G1_POINTGOAL_VIEWER_CONTROL_ENDPOINT) { $env:G1_POINTGOAL_VIEWER_CONTROL_ENDPOINT } else { "tcp://127.0.0.1:5580" }
    $ViewerTelemetryEndpoint = if ($env:G1_POINTGOAL_VIEWER_TELEMETRY_ENDPOINT) { $env:G1_POINTGOAL_VIEWER_TELEMETRY_ENDPOINT } else { "tcp://127.0.0.1:5581" }
    $ViewerShmName = if ($env:G1_POINTGOAL_VIEWER_SHM_NAME) { $env:G1_POINTGOAL_VIEWER_SHM_NAME } else { "g1_view_frames" }
    $ViewerShmSlotSize = if ($env:G1_POINTGOAL_VIEWER_SHM_SLOT_SIZE) { $env:G1_POINTGOAL_VIEWER_SHM_SLOT_SIZE } else { "8388608" }
    $ViewerShmCapacity = if ($env:G1_POINTGOAL_VIEWER_SHM_CAPACITY) { $env:G1_POINTGOAL_VIEWER_SHM_CAPACITY } else { "8" }
    $ViewerPublish = if ($env:G1_POINTGOAL_VIEWER_PUBLISH) { $env:G1_POINTGOAL_VIEWER_PUBLISH } else { "0" }
    $NativeViewer = if ($env:G1_POINTGOAL_NATIVE_VIEWER) { $env:G1_POINTGOAL_NATIVE_VIEWER } else { "off" }
    $ForceRuntimeCamera = if ($env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA) { $env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA } else { "0" }

    $EffectiveScenePreset = Get-LaunchArgValue -InputArgs $InputArgs -Names @("--scene-preset", "--scene") -DefaultValue $ScenePreset
    $EffectivePlannerMode = Get-LaunchArgValue -InputArgs $InputArgs -Names @("--planner-mode") -DefaultValue $PlannerMode
    $EffectiveLaunchMode = Get-LaunchArgValue -InputArgs $InputArgs -Names @("--launch-mode") -DefaultValue $LaunchMode
    $EffectiveNativeViewer = Get-LaunchArgValue -InputArgs $InputArgs -Names @("--native-viewer") -DefaultValue $NativeViewer
    $HasViewerPublish = Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-publish")
    $HasNoViewerPublish = Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--no-viewer-publish")
    $ForwardedArgs = Remove-LaunchArgs -InputArgs $InputArgs -Names @("--scene-preset", "--scene")
    $EffectiveViewerPublish = $ViewerPublish
    if ($HasViewerPublish) { $EffectiveViewerPublish = "1" }
    if ($HasNoViewerPublish) { $EffectiveViewerPublish = "0" }
    if ($EffectiveLaunchMode -eq "g1_view") {
        $EffectiveViewerPublish = "1"
        if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--native-viewer"))) {
            $EffectiveNativeViewer = "opencv"
        }
    }

    $SceneSelection = Resolve-SceneSelection -SelectedPreset $EffectiveScenePreset
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--policy")) -and -not (Test-Path -LiteralPath $PolicyPath)) {
        throw "runtime policy file not found: $PolicyPath"
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--robot_usd", "--robot-usd", "--usd-path")) -and -not (Test-Path -LiteralPath $RobotUsd)) {
        throw "runtime robot USD not found: $RobotUsd"
    }
    if (
        -not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--scene-usd", "--scene_usd", "--env-url")) -and
        -not [string]::IsNullOrWhiteSpace([string]$SceneSelection.SceneUsd) -and
        -not (Test-Path -LiteralPath ([string]$SceneSelection.SceneUsd))
    ) {
        throw "runtime scene USD not found: $([string]$SceneSelection.SceneUsd)"
    }
    $RuntimeArgs = @()

    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--policy"))) {
        $RuntimeArgs += @("--policy", $PolicyPath)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--robot_usd", "--robot-usd", "--usd-path"))) {
        $RuntimeArgs += @("--robot_usd", $RobotUsd)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--planner-mode"))) {
        $RuntimeArgs += @("--planner-mode", $EffectivePlannerMode)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--goal-x"))) {
        $RuntimeArgs += @("--goal-x", $GoalX)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--goal-y"))) {
        $RuntimeArgs += @("--goal-y", $GoalY)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--server-url"))) {
        $RuntimeArgs += @("--server-url", $ServerUrl)
    }
    if ($EffectivePlannerMode -eq "interactive" -and -not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--dual-server-url"))) {
        $RuntimeArgs += @("--dual-server-url", $DualServerUrl)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-control-endpoint"))) {
        $RuntimeArgs += @("--viewer-control-endpoint", $ViewerControlEndpoint)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-telemetry-endpoint"))) {
        $RuntimeArgs += @("--viewer-telemetry-endpoint", $ViewerTelemetryEndpoint)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-shm-name"))) {
        $RuntimeArgs += @("--viewer-shm-name", $ViewerShmName)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-shm-slot-size"))) {
        $RuntimeArgs += @("--viewer-shm-slot-size", $ViewerShmSlotSize)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-shm-capacity"))) {
        $RuntimeArgs += @("--viewer-shm-capacity", $ViewerShmCapacity)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--native-viewer"))) {
        $RuntimeArgs += @("--native-viewer", $EffectiveNativeViewer)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--viewer-publish", "--no-viewer-publish"))) {
        if ($EffectiveViewerPublish -in @("1", "true", "True", "TRUE", "yes", "Yes", "YES")) {
            $RuntimeArgs += @("--viewer-publish")
        }
        else {
            $RuntimeArgs += @("--no-viewer-publish")
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($EffectiveLaunchMode) -and -not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--launch-mode"))) {
        $RuntimeArgs += @("--launch-mode", $EffectiveLaunchMode)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--scene-usd", "--scene_usd", "--env-url"))) {
        if (-not [string]::IsNullOrWhiteSpace([string]$SceneSelection.SceneUsd)) {
            $RuntimeArgs += @("--scene-usd", [string]$SceneSelection.SceneUsd)
        }
        elseif (-not [string]::IsNullOrWhiteSpace([string]$SceneSelection.EnvUrl)) {
            $RuntimeArgs += @("--env-url", [string]$SceneSelection.EnvUrl)
        }
    }
    if ($ForceRuntimeCamera -notin @("0", "false", "False", "FALSE", "no", "No", "NO") -and -not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--force-runtime-camera"))) {
        $RuntimeArgs += @("--force-runtime-camera")
    }

    return @($RuntimeArgs + $ForwardedArgs)
}

function Invoke-NavComponent {
    param(
        [string[]]$Arguments = @()
    )

    $DefaultCheckpointCandidates = @(
        (Join-Path $RepoDir "artifacts\models\navdp-cross-modal.ckpt"),
        (Join-Path $RepoDir "navdp-weights.ckpt")
    )
    $Checkpoint = if ($env:NAVDP_CHECKPOINT) { $env:NAVDP_CHECKPOINT } else { ($DefaultCheckpointCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1) }
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        $Checkpoint = $DefaultCheckpointCandidates[0]
    }
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        throw "NavDP checkpoint not found: $Checkpoint"
    }

    Write-Host "[AURA_SYSTEM] starting Nav module on port $NavPort"
    $NavArgs = @("--port", [string]$NavPort, "--checkpoint", $Checkpoint) + $Arguments
    return Invoke-CondaModule -EntryModule $NavEntryModule -Arguments $NavArgs
}

function Invoke-S2Component {
    param(
        [string[]]$Arguments = @()
    )

    $ModelPath = if ($env:INTERNVLA_MODEL_PATH) { $env:INTERNVLA_MODEL_PATH } else { "artifacts/models/InternVLA-N1-System2.Q4_K_M.gguf" }
    $MmprojPath = if ($env:INTERNVLA_MMPROJ_PATH) { $env:INTERNVLA_MMPROJ_PATH } else { "artifacts/models/InternVLA-N1-System2.mmproj-Q8_0.gguf" }
    $ChatTemplateFile = if ($env:INTERNVLA_CHAT_TEMPLATE_FILE) { $env:INTERNVLA_CHAT_TEMPLATE_FILE } else { "scripts/internvla_system2_chat_template.jinja" }
    $LlamaServer = if ($env:LLAMA_SERVER_EXE) { $env:LLAMA_SERVER_EXE } else { "llama-server" }
    $ContextSize = if ($env:INTERNVLA_CTX_SIZE) { [int]$env:INTERNVLA_CTX_SIZE } else { 8192 }
    $GpuLayers = if ($env:INTERNVLA_GPU_LAYERS) { $env:INTERNVLA_GPU_LAYERS } else { "auto" }
    $PromptCache = if ($env:INTERNVLA_PROMPT_CACHE) { $env:INTERNVLA_PROMPT_CACHE } else { "off" }
    $CacheRamMiB = if ($env:INTERNVLA_CACHE_RAM_MIB) { [int]$env:INTERNVLA_CACHE_RAM_MIB } else { 0 }
    $Parallel = if ($env:INTERNVLA_PARALLEL) { [int]$env:INTERNVLA_PARALLEL } else { 1 }
    $ThreadsHttp = if ($env:INTERNVLA_THREADS_HTTP) { [int]$env:INTERNVLA_THREADS_HTTP } else { 1 }
    $ImageMaxTokens = if ($env:INTERNVLA_IMAGE_MAX_TOKENS) { [int]$env:INTERNVLA_IMAGE_MAX_TOKENS } else { 0 }
    $ReasoningBudget = if ($env:INTERNVLA_REASONING_BUDGET) { [int]$env:INTERNVLA_REASONING_BUDGET } else { 0 }

    $ResolvedModelPath = Resolve-ProjectPath $ModelPath
    $ResolvedMmprojPath = Resolve-ProjectPath $MmprojPath
    $ResolvedChatTemplateFile = Resolve-ProjectPath $ChatTemplateFile
    $ResolvedLlamaServerPath = Resolve-LlamaServerPath $LlamaServer

    if (-not (Test-Path -LiteralPath $ResolvedModelPath)) {
        throw "InternVLA model not found: $ResolvedModelPath"
    }
    if (-not (Test-Path -LiteralPath $ResolvedMmprojPath)) {
        throw "InternVLA mmproj not found: $ResolvedMmprojPath"
    }
    if (-not (Test-Path -LiteralPath $ResolvedChatTemplateFile)) {
        $ResolvedChatTemplateFile = $null
    }
    if ($null -eq $ResolvedLlamaServerPath -or -not (Test-Path -LiteralPath $ResolvedLlamaServerPath)) {
        throw "llama-server executable not found. Set LLAMA_SERVER_EXE."
    }
    if (-not (Test-LlamaGpuBackend -LlamaServerPath $ResolvedLlamaServerPath)) {
        throw "llama-server GPU backend not found near $ResolvedLlamaServerPath"
    }

    $ServerDir = Split-Path -Parent $ResolvedLlamaServerPath
    Add-ProcessPathEntry -PathEntry $ServerDir
    foreach ($RequiredDll in @("cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll")) {
        if (-not (Test-Path -LiteralPath (Join-Path $ServerDir $RequiredDll))) {
            throw "required CUDA runtime DLL not found near llama-server: $RequiredDll"
        }
    }

    $GpuLayerArg = if ($GpuLayers -eq "auto") { "-1" } else { $GpuLayers }
    $ServerArgs = @(
        "--host", $System2Host,
        "--port", [string]$System2Port,
        "--model", $ResolvedModelPath,
        "--mmproj", $ResolvedMmprojPath,
        "--ctx-size", [string]$ContextSize,
        "--gpu-layers", [string]$GpuLayerArg,
        "--cache-ram", [string]$CacheRamMiB,
        "--parallel", [string]$Parallel,
        "--threads-http", [string]$ThreadsHttp,
        "--reasoning-budget", [string]$ReasoningBudget
    )
    if ($PromptCache -eq "on") {
        $ServerArgs += @("--cache-prompt")
    }
    else {
        $ServerArgs += @("--no-cache-prompt")
    }
    if ($ImageMaxTokens -gt 0) {
        $ServerArgs += @("--image-max-tokens", [string]$ImageMaxTokens)
    }
    if ($null -ne $ResolvedChatTemplateFile) {
        $ServerArgs += @("--chat-template-file", $ResolvedChatTemplateFile)
    }

    Write-Host "[AURA_SYSTEM] starting S2 module on $System2Host`:$System2Port"
    Push-Location $RepoDir
    try {
        & $ResolvedLlamaServerPath @ServerArgs @Arguments
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Invoke-DualComponent {
    param(
        [string[]]$Arguments = @()
    )

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

    Write-Host "[AURA_SYSTEM] starting Main Control S2/Nav bridge on $DualHost`:$DualPort"
    $DualArgs = @(
        "--host", $DualHost,
        "--port", [string]$DualPort,
        "--navdp-url", $NavBaseUrl,
        "--vlm-url", $System2BaseUrl,
        "--vlm-model", $VLMModel,
        "--vlm-temperature", $VLMTemperature,
        "--vlm-top-k", $VLMTopK,
        "--vlm-top-p", $VLMTopP,
        "--vlm-min-p", $VLMMinP,
        "--vlm-repeat-penalty", $VLMRepeatPenalty,
        "--vlm-num-history", $VLMNumHistory,
        "--vlm-max-images-per-request", $VLMMaxImagesPerRequest,
        "--s2-mode", $S2Mode,
        "--vlm-timeout-sec", $VLMTimeoutSec,
        "--s2-failure-backoff-max-sec", $S2BackoffMaxSec
    ) + $Arguments
    return Invoke-CondaModule -EntryModule $DualEntryModule -Arguments $DualArgs
}

function Invoke-RuntimeComponent {
    param(
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path -LiteralPath $IsaacPython)) {
        throw "Isaac python launcher not found: $IsaacPython"
    }

    $RuntimeArgs = Build-RuntimeLaunchArgs -InputArgs $Arguments
    Write-Host "[AURA_SYSTEM] starting Main Control runtime via Isaac python"
    Push-Location $RepoDir
    $PreviousPythonPath = $env:PYTHONPATH
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        $env:PYTHONPATH = $SrcDir
    }
    else {
        $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
    }
    try {
        & $IsaacPython -m $RuntimeEntryModule @RuntimeArgs
        return $LASTEXITCODE
    }
    finally {
        if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHONPATH = $PreviousPythonPath
        }
        Pop-Location
    }
}

function Invoke-AllComponent {
    param(
        [string[]]$Arguments = @()
    )

    $PlannerMode = Get-LaunchArgValue -InputArgs $Arguments -Names @("--planner-mode") -DefaultValue (if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "interactive" })
    $Processes = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
    try {
        $NavProcess = Start-BackgroundSelf -Name "nav" -TargetComponent "nav"
        $Processes.Add($NavProcess) | Out-Null
        Wait-TcpReady -Host "127.0.0.1" -Port $NavPort -Name "Nav module" -TimeoutSec $StartupTimeoutSec -Process $NavProcess

        if ($PlannerMode -eq "interactive" -or $PlannerMode -eq "dual") {
            $S2Process = Start-BackgroundSelf -Name "s2" -TargetComponent "s2"
            $Processes.Add($S2Process) | Out-Null
            Wait-TcpReady -Host $System2Host -Port $System2Port -Name "S2 module" -TimeoutSec $StartupTimeoutSec -Process $S2Process

            $DualProcess = Start-BackgroundSelf -Name "dual" -TargetComponent "dual"
            $Processes.Add($DualProcess) | Out-Null
            Wait-TcpReady -Host $DualHost -Port $DualPort -Name "Dual bridge module" -TimeoutSec $StartupTimeoutSec -Process $DualProcess
        }

        return (Invoke-RuntimeComponent -Arguments $Arguments)
    }
    finally {
        for ($Index = $Processes.Count - 1; $Index -ge 0; $Index -= 1) {
            Stop-ManagedProcess -Process $Processes[$Index]
        }
    }
}

$ExitCode = 0
switch ($Component) {
    "nav" {
        $ExitCode = Invoke-NavComponent -Arguments $ForwardArgs
    }
    "s2" {
        $ExitCode = Invoke-S2Component -Arguments $ForwardArgs
    }
    "dual" {
        $ExitCode = Invoke-DualComponent -Arguments $ForwardArgs
    }
    "runtime" {
        $ExitCode = Invoke-RuntimeComponent -Arguments $ForwardArgs
    }
    "all" {
        $ExitCode = Invoke-AllComponent -Arguments $ForwardArgs
    }
}
exit $ExitCode
