$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "runtime.navigation_runtime"
$ViewerScript = Join-Path $ScriptDir "run_g1_viewer.ps1"
$ProcessLogDir = Join-Path $RepoDir "tmp\process_logs"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultIsaacPython = "C:\isaac-sim\python.bat"
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }

$DefaultPolicyCandidates = @(
    (Join-Path $RepoDir "artifacts\models\g1_policy_fp16.engine"),
    (Join-Path $RepoDir "artifacts\models\g1_policy_fp32.engine"),
    (Join-Path $RepoDir "artifacts\models\policy.onnx"),
    (Join-Path $RepoDir "policy.onnx"),
    (Join-Path $RepoDir "src\locomotion\models\policy_fp16.engine")
)
$ResolvedDefaultPolicy = $DefaultPolicyCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $ResolvedDefaultPolicy) {
    $ResolvedDefaultPolicy = $DefaultPolicyCandidates[0]
}
$PolicyPath = if ($env:G1_POINTGOAL_POLICY) {
    $env:G1_POINTGOAL_POLICY
} else {
    $ResolvedDefaultPolicy
}

$DefaultRobotUsd = Join-Path $RepoDir "src\locomotion\g1\g1_d455.usd"
$RobotUsd = if ($env:G1_POINTGOAL_ROBOT_USD) {
    $env:G1_POINTGOAL_ROBOT_USD
} else {
    $DefaultRobotUsd
}
$DefaultSceneEnvUrl = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
$DefaultInteriorAgentSceneUsd = Join-Path $RepoDir "datasets\InteriorAgent\kujiale_0004\kujiale_0004_navila_sanitized.usda"
$DefaultInteriorAgentKujiale3SceneUsd = Join-Path $RepoDir "datasets\InteriorAgent\kujiale_0003\kujiale_0003.usda"
$ScenePreset = if ($env:G1_POINTGOAL_SCENE_PRESET) { $env:G1_POINTGOAL_SCENE_PRESET } else { "warehouse" }
$SceneUsd = if ($env:G1_POINTGOAL_SCENE_USD) { $env:G1_POINTGOAL_SCENE_USD } else { "" }
$PlannerMode = if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "interactive" }
$GoalX = if ($env:G1_POINTGOAL_GOAL_X) { $env:G1_POINTGOAL_GOAL_X } else { "2.0" }
$GoalY = if ($env:G1_POINTGOAL_GOAL_Y) { $env:G1_POINTGOAL_GOAL_Y } else { "0.0" }
$LaunchMode = if ($env:G1_POINTGOAL_LAUNCH_MODE) { $env:G1_POINTGOAL_LAUNCH_MODE } else { "" }
$ServerUrl = if ($env:G1_POINTGOAL_SERVER_URL) { $env:G1_POINTGOAL_SERVER_URL } else { "http://127.0.0.1:8888" }
$ViewerControlEndpoint = if ($env:G1_POINTGOAL_VIEWER_CONTROL_ENDPOINT) { $env:G1_POINTGOAL_VIEWER_CONTROL_ENDPOINT } else { "tcp://127.0.0.1:5580" }
$ViewerTelemetryEndpoint = if ($env:G1_POINTGOAL_VIEWER_TELEMETRY_ENDPOINT) { $env:G1_POINTGOAL_VIEWER_TELEMETRY_ENDPOINT } else { "tcp://127.0.0.1:5581" }
$ViewerShmName = if ($env:G1_POINTGOAL_VIEWER_SHM_NAME) { $env:G1_POINTGOAL_VIEWER_SHM_NAME } else { "g1_view_frames" }
$ViewerShmSlotSize = if ($env:G1_POINTGOAL_VIEWER_SHM_SLOT_SIZE) { $env:G1_POINTGOAL_VIEWER_SHM_SLOT_SIZE } else { "8388608" }
$ViewerShmCapacity = if ($env:G1_POINTGOAL_VIEWER_SHM_CAPACITY) { $env:G1_POINTGOAL_VIEWER_SHM_CAPACITY } else { "8" }
$ViewerPublish = if ($env:G1_POINTGOAL_VIEWER_PUBLISH) { $env:G1_POINTGOAL_VIEWER_PUBLISH } else { "0" }
$NativeViewer = if ($env:G1_POINTGOAL_NATIVE_VIEWER) { $env:G1_POINTGOAL_NATIVE_VIEWER } else { "off" }
$ForceRuntimeCamera = if ($env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA) {
    $env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA
} else {
    "0"
}

function Test-LaunchArgPresent {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    foreach ($launchArg in $InputArgs) {
        foreach ($name in $Names) {
            if ($launchArg -eq $name -or $launchArg.StartsWith("$name=")) {
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

    $resolved = $DefaultValue
    for ($i = 0; $i -lt $InputArgs.Length; $i++) {
        $launchArg = $InputArgs[$i]
        foreach ($name in $Names) {
            if ($launchArg -eq $name) {
                if (($i + 1) -lt $InputArgs.Length) {
                    $resolved = $InputArgs[$i + 1]
                }
                continue
            }
            if ($launchArg.StartsWith("$name=")) {
                $resolved = $launchArg.Substring($name.Length + 1)
            }
        }
    }
    return $resolved
}

function Remove-LaunchArgs {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    $filtered = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $InputArgs.Length; $i++) {
        $launchArg = $InputArgs[$i]
        $matched = $false
        foreach ($name in $Names) {
            if ($launchArg -eq $name) {
                $matched = $true
                if (($i + 1) -lt $InputArgs.Length) {
                    $i += 1
                }
                break
            }
            if ($launchArg.StartsWith("$name=")) {
                $matched = $true
                break
            }
        }
        if (-not $matched) {
            $filtered.Add($launchArg)
        }
    }
    return $filtered.ToArray()
}

function Resolve-SceneSelection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedPreset,
        [Parameter(Mandatory = $true)]
        [string]$InteriorAgentSceneUsd,
        [Parameter(Mandatory = $true)]
        [string]$InteriorAgentKujiale3SceneUsd,
        [Parameter(Mandatory = $true)]
        [string]$WarehouseEnvUrl
    )

    $normalized = ("" + $SelectedPreset).Trim().ToLowerInvariant()
    switch ($normalized) {
        "" {
            return @{
                Preset = "warehouse"
                SceneUsd = ""
                EnvUrl = $WarehouseEnvUrl
                Description = "Isaac Simple_Warehouse"
            }
        }
        "warehouse" {
            return @{
                Preset = "warehouse"
                SceneUsd = ""
                EnvUrl = $WarehouseEnvUrl
                Description = "Isaac Simple_Warehouse"
            }
        }
        "interioragent" {
            return @{
                Preset = "interioragent"
                SceneUsd = $InteriorAgentSceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent"
            }
        }
        "interior" {
            return @{
                Preset = "interioragent"
                SceneUsd = $InteriorAgentSceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent"
            }
        }
        "datasets\\interioragent" {
            return @{
                Preset = "interioragent"
                SceneUsd = $InteriorAgentSceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent"
            }
        }
        "datasets/interioragent" {
            return @{
                Preset = "interioragent"
                SceneUsd = $InteriorAgentSceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent"
            }
        }
        "interior agent kujiale 3" {
            return @{
                Preset = "interior agent kujiale 3"
                SceneUsd = $InteriorAgentKujiale3SceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent\\kujiale_0003"
            }
        }
        "interioragent kujiale 3" {
            return @{
                Preset = "interior agent kujiale 3"
                SceneUsd = $InteriorAgentKujiale3SceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent\\kujiale_0003"
            }
        }
        "interioragent_kujiale3" {
            return @{
                Preset = "interior agent kujiale 3"
                SceneUsd = $InteriorAgentKujiale3SceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent\\kujiale_0003"
            }
        }
        "interioragent-kujiale3" {
            return @{
                Preset = "interior agent kujiale 3"
                SceneUsd = $InteriorAgentKujiale3SceneUsd
                EnvUrl = ""
                Description = "datasets\\InteriorAgent\\kujiale_0003"
            }
        }
        default {
            throw "unsupported scene preset: $SelectedPreset. Supported values: warehouse, interioragent, interior agent kujiale 3"
        }
    }
}

function Start-BackgroundPowerShell {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$ScriptArgs
    )

    New-Item -ItemType Directory -Path $ProcessLogDir -Force | Out-Null
    $safeName = ($Name -replace '[^A-Za-z0-9_-]', '_')
    $stdoutLog = Join-Path $ProcessLogDir "${safeName}.stdout.log"
    $stderrLog = Join-Path $ProcessLogDir "${safeName}.stderr.log"
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $ScriptPath
    ) + $ScriptArgs
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $RepoDir -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    Write-Host "[$Name] started pid=$($proc.Id)"
    Write-Host "[$Name] stdout-log=`"$stdoutLog`" stderr-log=`"$stderrLog`""
    return $proc
}

$HasPolicyOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--policy")
$HasRobotOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--robot_usd", "--robot-usd", "--usd-path")
$HasSceneOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--scene-usd", "--scene_usd", "--env-url")
$ForwardArgs = Remove-LaunchArgs -InputArgs $args -Names @("--scene-preset", "--scene")
$EffectiveScenePreset = Get-LaunchArgValue -InputArgs $args -Names @("--scene-preset", "--scene") -DefaultValue $ScenePreset
$EffectiveLaunchMode = Get-LaunchArgValue -InputArgs $args -Names @("--launch-mode") -DefaultValue $LaunchMode
$EffectivePlannerMode = Get-LaunchArgValue -InputArgs $args -Names @("--planner-mode") -DefaultValue $PlannerMode
$EffectiveViewerControlEndpoint = Get-LaunchArgValue -InputArgs $args -Names @("--viewer-control-endpoint") -DefaultValue $ViewerControlEndpoint
$EffectiveViewerTelemetryEndpoint = Get-LaunchArgValue -InputArgs $args -Names @("--viewer-telemetry-endpoint") -DefaultValue $ViewerTelemetryEndpoint
$EffectiveViewerShmName = Get-LaunchArgValue -InputArgs $args -Names @("--viewer-shm-name") -DefaultValue $ViewerShmName
$EffectiveViewerShmSlotSize = Get-LaunchArgValue -InputArgs $args -Names @("--viewer-shm-slot-size") -DefaultValue $ViewerShmSlotSize
$EffectiveViewerShmCapacity = Get-LaunchArgValue -InputArgs $args -Names @("--viewer-shm-capacity") -DefaultValue $ViewerShmCapacity
$EffectiveNativeViewer = Get-LaunchArgValue -InputArgs $args -Names @("--native-viewer") -DefaultValue $NativeViewer
$EffectiveDepthMaxM = Get-LaunchArgValue -InputArgs $args -Names @("--depth-max-m") -DefaultValue "5.0"
$ShowDepthView = Test-LaunchArgPresent -InputArgs $args -Names @("--show-depth")
$HasViewerPublish = Test-LaunchArgPresent -InputArgs $args -Names @("--viewer-publish")
$HasNoViewerPublish = Test-LaunchArgPresent -InputArgs $args -Names @("--no-viewer-publish")
$EffectiveViewerPublish = $ViewerPublish
if ($HasViewerPublish) {
    $EffectiveViewerPublish = "1"
}
if ($HasNoViewerPublish) {
    $EffectiveViewerPublish = "0"
}
if ($EffectiveLaunchMode -eq "g1_view") {
    $EffectiveViewerPublish = "1"
    if (-not (Test-LaunchArgPresent -InputArgs $args -Names @("--native-viewer"))) {
        $EffectiveNativeViewer = "opencv"
    }
}
$ViewerPublishEnabled = $EffectiveViewerPublish -notin @("0", "false", "False", "FALSE", "no", "No", "NO")

$ResolvedScene = Resolve-SceneSelection `
    -SelectedPreset $EffectiveScenePreset `
    -InteriorAgentSceneUsd $DefaultInteriorAgentSceneUsd `
    -InteriorAgentKujiale3SceneUsd $DefaultInteriorAgentKujiale3SceneUsd `
    -WarehouseEnvUrl $DefaultSceneEnvUrl

$DefaultSceneUsd = [string]$ResolvedScene.SceneUsd
$DefaultEnvUrl = [string]$ResolvedScene.EnvUrl
$EffectiveSceneDescription = [string]$ResolvedScene.Description

if ($EffectivePlannerMode -notin @("interactive", "pointgoal")) {
    Write-Host "[AURA Runtime] unsupported planner-mode=$EffectivePlannerMode"
    Write-Host "[AURA Runtime] supported planner modes: interactive, pointgoal"
    exit 1
}

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    Write-Host "[AURA Runtime] Isaac python launcher not found: `"$IsaacPython`""
    Write-Host "[AURA Runtime] Set ISAAC_SIM_PYTHON to your python.bat path."
    exit 1
}

if ((-not $HasPolicyOverride) -and (-not (Test-Path -LiteralPath $PolicyPath))) {
    Write-Host "[AURA Runtime] policy file not found: `"$PolicyPath`""
    exit 1
}

if ((-not $HasRobotOverride) -and (-not (Test-Path -LiteralPath $RobotUsd))) {
    Write-Host "[AURA Runtime] G1 USD not found: `"$RobotUsd`""
    exit 1
}

if (-not $HasSceneOverride) {
    if (-not [string]::IsNullOrWhiteSpace($SceneUsd)) {
        if (-not (Test-Path -LiteralPath $SceneUsd)) {
            Write-Host "[AURA Runtime] Scene USD not found: `"$SceneUsd`""
            Write-Host "[AURA Runtime] Set G1_POINTGOAL_SCENE_USD, choose --scene-preset warehouse|interioragent|\"interior agent kujiale 3\", or pass --scene-usd/--env-url explicitly."
            exit 1
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($DefaultSceneUsd) -and (-not (Test-Path -LiteralPath $DefaultSceneUsd))) {
        Write-Host "[AURA Runtime] Scene preset path not found for ${EffectiveScenePreset}: `"$DefaultSceneUsd`""
        exit 1
    }
}

Write-Host "[AURA Runtime] Starting runtime.navigation_runtime"
Write-Host "[AURA Runtime] python=`"$IsaacPython`""
Write-Host "[AURA Runtime] module=`"$EntryModule`""
Write-Host "[AURA Runtime] default policy=`"$PolicyPath`""
Write-Host "[AURA Runtime] default robot-usd=`"$RobotUsd`""
Write-Host "[AURA Runtime] default scene-preset=$ScenePreset"
Write-Host "[AURA Runtime] effective scene-preset=$EffectiveScenePreset"
if (-not [string]::IsNullOrWhiteSpace($SceneUsd)) {
    Write-Host "[AURA Runtime] default scene-usd=`"$SceneUsd`""
} elseif (-not [string]::IsNullOrWhiteSpace($DefaultSceneUsd)) {
    Write-Host "[AURA Runtime] default scene-usd=`"$DefaultSceneUsd`" ($EffectiveSceneDescription)"
} elseif (-not [string]::IsNullOrWhiteSpace($DefaultEnvUrl)) {
    Write-Host "[AURA Runtime] default env-url=`"$DefaultEnvUrl`" ($EffectiveSceneDescription)"
} else {
    Write-Host "[AURA Runtime] default scene-usd=<flat ground only>"
}
Write-Host "[AURA Runtime] default planner-mode=$PlannerMode"
Write-Host "[AURA Runtime] effective planner-mode=$EffectivePlannerMode"
Write-Host "[AURA Runtime] default goal=($GoalX, $GoalY)"
if ([string]::IsNullOrWhiteSpace($LaunchMode)) {
    Write-Host "[AURA Runtime] default launch-mode=<legacy>"
} else {
    Write-Host "[AURA Runtime] default launch-mode=$LaunchMode"
}
Write-Host "[AURA Runtime] default server-url=$ServerUrl"
Write-Host "[AURA Runtime] default viewer-control-endpoint=$ViewerControlEndpoint"
Write-Host "[AURA Runtime] default viewer-telemetry-endpoint=$ViewerTelemetryEndpoint"
Write-Host "[AURA Runtime] default viewer-shm-name=$ViewerShmName"
Write-Host "[AURA Runtime] default viewer-publish=$ViewerPublish"
Write-Host "[AURA Runtime] effective viewer-publish=$EffectiveViewerPublish"
Write-Host "[AURA Runtime] effective native-viewer=$EffectiveNativeViewer"
Write-Host "[AURA Runtime] default force-runtime-camera=$ForceRuntimeCamera"
Write-Host "[AURA Runtime] default memory-store=on"
Write-Host "[AURA Runtime] default detection=on"
Write-Host "[AURA Runtime] user args override defaults when repeated."
Write-Host "[AURA Runtime] examples:"
Write-Host "[AURA Runtime]   interactive: .\\run_aura_runtime.ps1 --planner-mode interactive --launch-mode gui"
Write-Host "[AURA Runtime]   warehouse  : .\\run_aura_runtime.ps1 --scene-preset warehouse --planner-mode interactive"
Write-Host "[AURA Runtime]   interior   : .\\run_aura_runtime.ps1 --scene-preset interioragent --planner-mode interactive"
Write-Host "[AURA Runtime]   kujiale 3  : .\\run_aura_runtime.ps1 --scene-preset \"interior agent kujiale 3\" --planner-mode interactive"
Write-Host "[AURA Runtime]   pointgoal  : .\\run_aura_runtime.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0"
Write-Host "[AURA Runtime]   pointgoal+v: .\\run_aura_runtime.ps1 --planner-mode pointgoal --launch-mode g1_view --goal-x 2.0 --goal-y 0.0"
Write-Host "[AURA Runtime]   interac+d  : .\\run_aura_runtime.ps1 --planner-mode interactive --launch-mode g1_view --show-depth"
Write-Host "[AURA Runtime]   no memory : .\\run_aura_runtime.ps1 --no-memory-store"
Write-Host "[AURA Runtime]   no detect: .\\run_aura_runtime.ps1 --skip-detection"

$LaunchArgs = @(
    "-m", $EntryModule,
    "--policy", $PolicyPath,
    "--robot_usd", $RobotUsd,
    "--planner-mode", $EffectivePlannerMode,
    "--goal-x", $GoalX,
    "--goal-y", $GoalY,
    "--server-url", $ServerUrl,
    "--viewer-control-endpoint", $ViewerControlEndpoint,
    "--viewer-telemetry-endpoint", $ViewerTelemetryEndpoint,
    "--viewer-shm-name", $ViewerShmName,
    "--viewer-shm-slot-size", $ViewerShmSlotSize,
    "--viewer-shm-capacity", $ViewerShmCapacity,
    "--native-viewer", $EffectiveNativeViewer
)

if ($ViewerPublishEnabled) {
    $LaunchArgs += @("--viewer-publish")
} else {
    $LaunchArgs += @("--no-viewer-publish")
}

if (-not [string]::IsNullOrWhiteSpace($EffectiveLaunchMode)) {
    $LaunchArgs += @("--launch-mode", $EffectiveLaunchMode)
}

if (-not $HasSceneOverride -and -not [string]::IsNullOrWhiteSpace($SceneUsd)) {
    $LaunchArgs += @("--scene-usd", $SceneUsd)
} elseif (-not $HasSceneOverride -and -not [string]::IsNullOrWhiteSpace($DefaultSceneUsd)) {
    $LaunchArgs += @("--scene-usd", $DefaultSceneUsd)
} elseif (-not $HasSceneOverride -and -not [string]::IsNullOrWhiteSpace($DefaultEnvUrl)) {
    $LaunchArgs += @("--env-url", $DefaultEnvUrl)
}

if ($ForceRuntimeCamera -notin @("0", "false", "False", "FALSE", "no", "No", "NO")) {
    $LaunchArgs += @("--force-runtime-camera")
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
$ViewerProcess = $null
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    if ($ViewerPublishEnabled -and $EffectiveNativeViewer -eq "opencv") {
        if (-not (Test-Path -LiteralPath $ViewerScript)) {
            Write-Host "[AURA Runtime] viewer launcher not found: `"$ViewerScript`""
            exit 1
        }
        $ViewerArgs = @(
            "--control-endpoint", $EffectiveViewerControlEndpoint,
            "--telemetry-endpoint", $EffectiveViewerTelemetryEndpoint,
            "--shm-name", $EffectiveViewerShmName,
            "--shm-slot-size", $EffectiveViewerShmSlotSize,
            "--shm-capacity", $EffectiveViewerShmCapacity,
            "--depth-max-m", $EffectiveDepthMaxM
        )
        if ($ShowDepthView) {
            $ViewerArgs += @("--show-depth")
        }
        $ViewerProcess = Start-BackgroundPowerShell -ScriptPath $ViewerScript -Name "G1Viewer" -ScriptArgs $ViewerArgs
    }
    & $IsaacPython @LaunchArgs @ForwardArgs
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $ViewerProcess) {
        try {
            if (-not $ViewerProcess.HasExited) {
                Stop-Process -Id $ViewerProcess.Id -Force
            }
        } catch {
        }
    }
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Pop-Location
}
