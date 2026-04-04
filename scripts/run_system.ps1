param(
    [ValidateSet("all", "nav", "s2", "runtime")]
    [string]$Component = "all",
    [int]$StartupTimeoutSec = 180,
    [switch]$PrintConfigJson,
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

function Get-PreferredEnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        [string]$DefaultValue = ""
    )

    foreach ($Name in $Names) {
        $Value = [Environment]::GetEnvironmentVariable($Name, "Process")
        if (-not [string]::IsNullOrWhiteSpace($Value)) {
            return [string]$Value
        }
    }
    return [string]$DefaultValue
}

function Resolve-PortFromUrl {
    param(
        [string]$Url,
        [int]$DefaultPort
    )

    if ([string]::IsNullOrWhiteSpace($Url)) {
        return [int]$DefaultPort
    }
    try {
        $Parsed = [Uri]$Url
        if ($Parsed.Port -gt 0) {
            return [int]$Parsed.Port
        }
    }
    catch {
    }
    return [int]$DefaultPort
}

function Resolve-CondaExeFromBat {
    param(
        [string]$CondaBatPath
    )

    if ([string]::IsNullOrWhiteSpace($CondaBatPath)) {
        return $null
    }
    try {
        $ResolvedBat = [System.IO.Path]::GetFullPath($CondaBatPath)
    }
    catch {
        return $null
    }
    $BatDir = Split-Path -Parent $ResolvedBat
    $Candidates = @(
        (Join-Path $BatDir "conda.exe"),
        (Join-Path $BatDir "..\Scripts\conda.exe"),
        (Join-Path $BatDir "..\conda.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return [System.IO.Path]::GetFullPath($Candidate)
        }
    }
    return $null
}

$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$DefaultCondaEnv = "fa2-cu130-py312"
$DefaultIsaacPython = "C:\isaac-sim\python.bat"

$NavEntryModule = "apps.navdp_server_app"
$System2EntryModule = "apps.system2_app"
$RuntimeEntryModule = "runtime.aura_runtime"

$NavBaseUrl = Get-PreferredEnvValue -Names @("NAVDP_URL", "G1_POINTGOAL_SERVER_URL") -DefaultValue "http://127.0.0.1:8888"
$NavPort = if ($env:NAVDP_PORT) { [int]$env:NAVDP_PORT } else { Resolve-PortFromUrl -Url $NavBaseUrl -DefaultPort 8888 }
$System2BaseUrl = Get-PreferredEnvValue -Names @("INTERNVLA_URL", "G1_POINTGOAL_SYSTEM2_URL") -DefaultValue "http://127.0.0.1:15801"
$System2Host = if ($env:INTERNVLA_HOST) { $env:INTERNVLA_HOST } else { try { ([Uri]$System2BaseUrl).Host } catch { "127.0.0.1" } }
$System2Port = if ($env:INTERNVLA_PORT) { [int]$env:INTERNVLA_PORT } else { Resolve-PortFromUrl -Url $System2BaseUrl -DefaultPort 15801 }
$NavBaseUrl = "http://127.0.0.1:$NavPort"
$System2BaseUrl = "http://$System2Host`:$System2Port"
$CondaBat = Get-PreferredEnvValue -Names @("CONDA_BAT") -DefaultValue ""
$CondaExe = Get-PreferredEnvValue -Names @("CONDA_EXE", "AURA_CONDA_EXE") -DefaultValue ""
if ([string]::IsNullOrWhiteSpace($CondaExe) -and -not [string]::IsNullOrWhiteSpace($CondaBat)) {
    $CondaExe = Resolve-CondaExeFromBat -CondaBatPath $CondaBat
}
if ([string]::IsNullOrWhiteSpace($CondaExe)) {
    $CondaExe = $DefaultCondaExe
}
$CondaEnv = Get-PreferredEnvValue -Names @("CONDA_ENV_NAME", "AURA_CONDA_ENV") -DefaultValue $DefaultCondaEnv
$IsaacSimRoot = Get-PreferredEnvValue -Names @("ISAACSIM_PATH") -DefaultValue ""
$IsaacPython = if (-not [string]::IsNullOrWhiteSpace($IsaacSimRoot)) { Join-Path $IsaacSimRoot "python.bat" } elseif ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }

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

function Get-ProcessCommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
        if ($null -eq $ProcessInfo) {
            return ""
        }
        return [string]$ProcessInfo.CommandLine
    }
    catch {
        return ""
    }
}

function Resolve-LocalTcpEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Endpoint
    )

    $Trimmed = [string]$Endpoint
    if ([string]::IsNullOrWhiteSpace($Trimmed)) {
        return @{ IsLocal = $false; Port = 0 }
    }
    $Trimmed = $Trimmed.Trim()
    if ($Trimmed -notmatch '^tcp://(?<Host>\[[^\]]+\]|[^:]+):(?<Port>\d+)$') {
        return @{ IsLocal = $false; Port = 0 }
    }

    $HostValue = [string]$Matches["Host"]
    if ($HostValue.StartsWith("[") -and $HostValue.EndsWith("]")) {
        $HostValue = $HostValue.Substring(1, $HostValue.Length - 2)
    }
    $NormalizedHost = $HostValue.Trim().ToLowerInvariant()
    $IsLocal = $NormalizedHost -in @("127.0.0.1", "localhost", "0.0.0.0", "::1")
    return @{
        IsLocal = $IsLocal
        Port    = [int]$Matches["Port"]
    }
}

function Stop-StaleRuntimeBridgeListeners {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoDir,
        [string[]]$Endpoints = @()
    )

    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
        return
    }

    $ExistingPids = @()
    $TrackedPorts = @()
    foreach ($Endpoint in @($Endpoints)) {
        $Binding = Resolve-LocalTcpEndpoint -Endpoint ([string]$Endpoint)
        if (-not $Binding.IsLocal) {
            continue
        }
        $TrackedPorts += [int]$Binding.Port
        $ExistingPids += @(
            Get-NetTCPConnection -State Listen -LocalPort ([int]$Binding.Port) -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique
        )
    }

    $TrackedPorts = @($TrackedPorts | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    $ExistingPids = @($ExistingPids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($TrackedPorts.Count -eq 0 -or $ExistingPids.Count -eq 0) {
        return
    }

    foreach ($ProcessId in $ExistingPids) {
        $CommandLine = Get-ProcessCommandLine -ProcessId ([int]$ProcessId)
        $IsRuntimeProcess = $CommandLine -like "*runtime.aura_runtime*" -and $CommandLine -like "*isaac-aura*"
        if (-not $IsRuntimeProcess) {
            throw "Runtime bridge port(s) $([string]::Join(',', $TrackedPorts)) are already in use by pid=$ProcessId. Refusing to stop a non-runtime process."
        }
        Write-Host "[AURA_SYSTEM] stopping stale runtime bridge pid=$ProcessId on port(s) $([string]::Join(',', $TrackedPorts))"
        Stop-Process -Id ([int]$ProcessId) -Force -ErrorAction Stop
    }

    $Deadline = (Get-Date).AddSeconds(10)
    do {
        $StillListening = $false
        foreach ($TrackedPort in $TrackedPorts) {
            $ActiveConnections = @(
                Get-NetTCPConnection -State Listen -LocalPort ([int]$TrackedPort) -ErrorAction SilentlyContinue
            )
            if ($ActiveConnections.Count -gt 0) {
                $StillListening = $true
                break
            }
        }
        if (-not $StillListening) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $Deadline)

    throw "Stale runtime bridge listener on port(s) $([string]::Join(',', $TrackedPorts)) did not exit in time."
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
    $PlannerMode = if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "IDLE" }
    $LaunchMode = if ($env:G1_POINTGOAL_LAUNCH_MODE) { $env:G1_POINTGOAL_LAUNCH_MODE } else { "" }
    $ServerUrl = Get-PreferredEnvValue -Names @("NAVDP_URL", "G1_POINTGOAL_SERVER_URL") -DefaultValue $NavBaseUrl
    $System2Url = Get-PreferredEnvValue -Names @("INTERNVLA_URL", "G1_POINTGOAL_SYSTEM2_URL") -DefaultValue $System2BaseUrl
    $Instruction = Get-PreferredEnvValue -Names @("NAV_INSTRUCTION") -DefaultValue ""
    $InstructionLanguage = Get-PreferredEnvValue -Names @("NAV_INSTRUCTION_LANGUAGE") -DefaultValue "auto"
    $NavCommandApiHost = Get-PreferredEnvValue -Names @("NAV_COMMAND_API_HOST") -DefaultValue "127.0.0.1"
    $NavCommandApiPort = Get-PreferredEnvValue -Names @("NAV_COMMAND_API_PORT") -DefaultValue "8892"
    $CameraApiHost = Get-PreferredEnvValue -Names @("CAMERA_API_HOST") -DefaultValue "127.0.0.1"
    $CameraApiPort = Get-PreferredEnvValue -Names @("CAMERA_API_PORT") -DefaultValue "8891"
    $CameraPitchDeg = Get-PreferredEnvValue -Names @("CAMERA_PITCH_DEG") -DefaultValue "0.0"
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
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--server-url", "--server_url", "--navdp-url", "--navdp_url"))) {
        $RuntimeArgs += @("--server-url", $ServerUrl)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--system2-url", "--internvla-url", "--system2_url", "--internvla_url"))) {
        $RuntimeArgs += @("--system2-url", $System2Url)
    }
    if (-not [string]::IsNullOrWhiteSpace($Instruction) -and -not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--instruction", "--nav-instruction", "--nav_instruction"))) {
        $RuntimeArgs += @("--nav-instruction", $Instruction)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--nav-instruction-language", "--nav_instruction_language"))) {
        $RuntimeArgs += @("--nav-instruction-language", $InstructionLanguage)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--nav-command-api-host", "--nav_command_api_host"))) {
        $RuntimeArgs += @("--nav-command-api-host", $NavCommandApiHost)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--nav-command-api-port", "--nav_command_api_port"))) {
        $RuntimeArgs += @("--nav-command-api-port", $NavCommandApiPort)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--camera-api-host", "--camera_api_host"))) {
        $RuntimeArgs += @("--camera-api-host", $CameraApiHost)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--camera-api-port", "--camera_api_port"))) {
        $RuntimeArgs += @("--camera-api-port", $CameraApiPort)
    }
    if (-not (Test-LaunchArgPresent -InputArgs $InputArgs -Names @("--camera-pitch-deg", "--camera_pitch_deg"))) {
        $RuntimeArgs += @("--camera-pitch-deg", $CameraPitchDeg)
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

    $ModelPath = if ($env:INTERNVLA_LLAMA_MODEL_PATH) { $env:INTERNVLA_LLAMA_MODEL_PATH } elseif ($env:INTERNVLA_MODEL_PATH) { $env:INTERNVLA_MODEL_PATH } else { "artifacts/models/InternVLA-N1-System2.Q4_K_M.gguf" }
    $MmprojPath = if ($env:INTERNVLA_LLAMA_MMPROJ_PATH) { $env:INTERNVLA_LLAMA_MMPROJ_PATH } elseif ($env:INTERNVLA_MMPROJ_PATH) { $env:INTERNVLA_MMPROJ_PATH } else { "artifacts/models/InternVLA-N1-System2.mmproj-Q8_0.gguf" }
    $PromptModelPath = if ($env:INTERNVLA_PROMPT_MODEL_PATH) { $env:INTERNVLA_PROMPT_MODEL_PATH } else { "" }
    $LlamaServer = if ($env:LLAMA_SERVER_EXE) { $env:LLAMA_SERVER_EXE } else { "llama-server" }
    $ContextSize = if ($env:INTERNVLA_CTX_SIZE) { [int]$env:INTERNVLA_CTX_SIZE } else { 8192 }
    $GpuLayers = if ($env:INTERNVLA_GPU_LAYERS) { $env:INTERNVLA_GPU_LAYERS } else { "auto" }
    $PromptCache = if ($env:INTERNVLA_PROMPT_CACHE) { $env:INTERNVLA_PROMPT_CACHE } else { "off" }
    $CacheTypeK = if ($env:INTERNVLA_LLAMA_CACHE_TYPE_K) { $env:INTERNVLA_LLAMA_CACHE_TYPE_K } else { "q8_0" }
    $CacheTypeV = if ($env:INTERNVLA_LLAMA_CACHE_TYPE_V) { $env:INTERNVLA_LLAMA_CACHE_TYPE_V } else { "q8_0" }
    $NumHistory = if ($env:INTERNVLA_NUM_HISTORY) { [int]$env:INTERNVLA_NUM_HISTORY } else { 4 }
    $PlanStepGap = if ($env:INTERNVLA_PLAN_STEP_GAP) { [int]$env:INTERNVLA_PLAN_STEP_GAP } else { 4 }
    $LlamaThreads = if ($env:INTERNVLA_LLAMA_THREADS) { $env:INTERNVLA_LLAMA_THREADS } else { "" }
    $MainGpu = if ($env:INTERNVLA_LLAMA_MAIN_GPU) { [int]$env:INTERNVLA_LLAMA_MAIN_GPU } else { 0 }
    $LlamaFlashAttn = if ($env:INTERNVLA_LLAMA_FLASH_ATTN) { $env:INTERNVLA_LLAMA_FLASH_ATTN } else { "on" }
    $ChatLoraPath = if ($env:INTERNVLA_LLAMA_CHAT_LORA_PATH) { $env:INTERNVLA_LLAMA_CHAT_LORA_PATH } else { "" }
    $ChatLoraScale = if ($env:INTERNVLA_LLAMA_CHAT_LORA_SCALE) { $env:INTERNVLA_LLAMA_CHAT_LORA_SCALE } else { "1.0" }
    $ChatSystemPrompt = if ($env:INTERNVLA_CHAT_SESSION_SYSTEM_PROMPT) { $env:INTERNVLA_CHAT_SESSION_SYSTEM_PROMPT } else { "" }
    $LlamaUrl = if ($env:INTERNVLA_LLAMA_URL) { $env:INTERNVLA_LLAMA_URL } else { "http://127.0.0.1:$($System2Port + 1)" }

    $ResolvedModelPath = Resolve-ProjectPath $ModelPath
    $ResolvedMmprojPath = Resolve-ProjectPath $MmprojPath
    $ResolvedPromptModelPath = $null
    if (-not [string]::IsNullOrWhiteSpace($PromptModelPath)) {
        $ResolvedPromptModelPath = Resolve-ProjectPath $PromptModelPath
    }
    $ResolvedLlamaServerPath = Resolve-LlamaServerPath $LlamaServer

    if (-not (Test-Path -LiteralPath $ResolvedModelPath)) {
        throw "InternVLA model not found: $ResolvedModelPath"
    }
    if (-not (Test-Path -LiteralPath $ResolvedMmprojPath)) {
        throw "InternVLA mmproj not found: $ResolvedMmprojPath"
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

    Write-Host "[AURA_SYSTEM] starting S2 module on $System2Host`:$System2Port"
    $GpuLayerArg = if ($GpuLayers -eq "auto") { "-1" } else { $GpuLayers }
    $WrapperArgs = @(
        "--host", $System2Host,
        "--port", [string]$System2Port,
        "--llama-cpp-root", (Split-Path -Parent $ResolvedLlamaServerPath),
        "--llama-server-path", $ResolvedLlamaServerPath,
        "--llama-model-path", $ResolvedModelPath,
        "--llama-mmproj-path", $ResolvedMmprojPath,
        "--llama-url", $LlamaUrl,
        "--llama-ctx-size", [string]$ContextSize,
        "--llama-gpu-layers", [string]$GpuLayerArg,
        "--llama-main-gpu", [string]$MainGpu,
        "--llama-flash-attn", $LlamaFlashAttn,
        "--llama-cache-type-k", $CacheTypeK,
        "--llama-cache-type-v", $CacheTypeV,
        "--llama-cache-prompt", $PromptCache,
        "--num-history", [string]$NumHistory,
        "--plan-step-gap", [string]$PlanStepGap
    )
    if (-not [string]::IsNullOrWhiteSpace($LlamaThreads)) {
        $WrapperArgs += @("--llama-threads", $LlamaThreads)
    }
    if ($null -ne $ResolvedPromptModelPath -and (Test-Path -LiteralPath $ResolvedPromptModelPath)) {
        $WrapperArgs += @("--prompt-model-path", $ResolvedPromptModelPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($ChatLoraPath)) {
        $ResolvedChatLoraPath = Resolve-ProjectPath $ChatLoraPath
        $WrapperArgs += @("--llama-chat-lora-path", $ResolvedChatLoraPath, "--llama-chat-lora-scale", $ChatLoraScale)
    }
    if (-not [string]::IsNullOrWhiteSpace($ChatSystemPrompt)) {
        $WrapperArgs += @("--chat-session-system-prompt", $ChatSystemPrompt)
    }
    return Invoke-CondaModule -EntryModule $System2EntryModule -Arguments ($WrapperArgs + $Arguments)
}

function Invoke-RuntimeComponent {
    param(
        [string[]]$Arguments = @()
    )

    if (-not (Test-Path -LiteralPath $IsaacPython)) {
        throw "Isaac python launcher not found: $IsaacPython"
    }

    $RuntimeArgs = Build-RuntimeLaunchArgs -InputArgs $Arguments
    $ViewerControlEndpoint = Get-LaunchArgValue -InputArgs $RuntimeArgs -Names @("--viewer-control-endpoint") -DefaultValue ""
    $ViewerTelemetryEndpoint = Get-LaunchArgValue -InputArgs $RuntimeArgs -Names @("--viewer-telemetry-endpoint") -DefaultValue ""
    Stop-StaleRuntimeBridgeListeners -RepoDir $RepoDir -Endpoints @($ViewerControlEndpoint, $ViewerTelemetryEndpoint)
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

    $HelperScript = Join-Path $ScriptDir "run_windows_fullstack.ps1"
    if (-not (Test-Path -LiteralPath $HelperScript)) {
        throw "Windows full-stack helper not found: $HelperScript"
    }
    if ($PrintConfigJson) {
        & $HelperScript -StartupTimeoutSec $StartupTimeoutSec -NoExit -PrintConfigJson @Arguments
    }
    else {
        & $HelperScript -StartupTimeoutSec $StartupTimeoutSec -NoExit @Arguments
    }
    if ($null -ne $LASTEXITCODE) {
        return [int]$LASTEXITCODE
    }
    return 0
}

$ExitCode = 0
switch ($Component) {
    "nav" {
        $ExitCode = Invoke-NavComponent -Arguments $ForwardArgs
    }
    "s2" {
        $ExitCode = Invoke-S2Component -Arguments $ForwardArgs
    }
    "runtime" {
        $ExitCode = Invoke-RuntimeComponent -Arguments $ForwardArgs
    }
    "all" {
        $ExitCode = Invoke-AllComponent -Arguments $ForwardArgs
    }
}
exit $ExitCode
