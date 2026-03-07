$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "runtime.g1_bridge"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultIsaacPython = "C:\isaac-sim\python.bat"
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }

$DefaultPolicyCandidates = @(
    (Join-Path $RepoDir "artifacts\models\policy.onnx"),
    (Join-Path $RepoDir "policy.onnx")
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
$SceneUsd = if ($env:G1_POINTGOAL_SCENE_USD) { $env:G1_POINTGOAL_SCENE_USD } else { "/Isaac/Environments/Simple_Warehouse/warehouse.usd" }
$PlannerMode = if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "interactive" }
$GoalX = if ($env:G1_POINTGOAL_GOAL_X) { $env:G1_POINTGOAL_GOAL_X } else { "2.0" }
$GoalY = if ($env:G1_POINTGOAL_GOAL_Y) { $env:G1_POINTGOAL_GOAL_Y } else { "0.0" }
$Instruction = if ($env:G1_POINTGOAL_INSTRUCTION) {
    $env:G1_POINTGOAL_INSTRUCTION
} else {
    "Navigate safely to the target and stop when complete."
}
$ServerUrl = if ($env:G1_POINTGOAL_SERVER_URL) { $env:G1_POINTGOAL_SERVER_URL } else { "http://127.0.0.1:8888" }
$DualServerUrl = if ($env:G1_POINTGOAL_DUAL_SERVER_URL) { $env:G1_POINTGOAL_DUAL_SERVER_URL } else { "http://127.0.0.1:8890" }
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

$HasPolicyOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--policy")
$HasRobotOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--robot_usd", "--robot-usd", "--usd-path")
$HasSceneOverride = Test-LaunchArgPresent -InputArgs $args -Names @("--scene-usd", "--scene_usd", "--env-url")

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    Write-Host "[G1 PointGoal] Isaac python launcher not found: `"$IsaacPython`""
    Write-Host "[G1 PointGoal] Set ISAAC_SIM_PYTHON to your python.bat path."
    exit 1
}

if ((-not $HasPolicyOverride) -and (-not (Test-Path -LiteralPath $PolicyPath))) {
    Write-Host "[G1 PointGoal] ONNX policy not found: `"$PolicyPath`""
    exit 1
}

if ((-not $HasRobotOverride) -and (-not (Test-Path -LiteralPath $RobotUsd))) {
    Write-Host "[G1 PointGoal] G1 USD not found: `"$RobotUsd`""
    exit 1
}

if (
    (-not $HasSceneOverride) -and
    -not [string]::IsNullOrWhiteSpace($SceneUsd) -and
    ($SceneUsd -match '^[A-Za-z]:[\\/]') -and
    (-not (Test-Path -LiteralPath $SceneUsd))
) {
    Write-Host "[G1 PointGoal] Scene USD not found: `"$SceneUsd`""
    Write-Host "[G1 PointGoal] Set G1_POINTGOAL_SCENE_USD or pass --scene-usd/--env-url explicitly."
    exit 1
}

Write-Host "[G1 PointGoal] Starting NavDP locomotion bridge"
Write-Host "[G1 PointGoal] python=`"$IsaacPython`""
Write-Host "[G1 PointGoal] module=`"$EntryModule`""
Write-Host "[G1 PointGoal] default policy=`"$PolicyPath`""
Write-Host "[G1 PointGoal] default robot-usd=`"$RobotUsd`""
if ([string]::IsNullOrWhiteSpace($SceneUsd)) {
    Write-Host "[G1 PointGoal] default scene-usd=<flat ground only>"
} else {
    Write-Host "[G1 PointGoal] default scene-usd=`"$SceneUsd`""
}
Write-Host "[G1 PointGoal] default planner-mode=$PlannerMode"
Write-Host "[G1 PointGoal] default goal=($GoalX, $GoalY)"
Write-Host "[G1 PointGoal] default instruction=`"$Instruction`""
Write-Host "[G1 PointGoal] default server-url=$ServerUrl"
Write-Host "[G1 PointGoal] default dual-server-url=$DualServerUrl"
Write-Host "[G1 PointGoal] default force-runtime-camera=$ForceRuntimeCamera"
Write-Host "[G1 PointGoal] user args override defaults when repeated."
Write-Host "[G1 PointGoal] examples:"
Write-Host "[G1 PointGoal]   interactive: .\\run_g1_pointgoal.ps1 --planner-mode interactive"
Write-Host "[G1 PointGoal]   pointgoal  : .\\run_g1_pointgoal.ps1 --planner-mode pointgoal --goal-x 2.0 --goal-y 0.0"
Write-Host "[G1 PointGoal]   dual       : .\\run_g1_pointgoal.ps1 --planner-mode dual --instruction `"Navigate to the target and stop.`""

$LaunchArgs = @(
    "-m", $EntryModule,
    "--policy", $PolicyPath,
    "--robot_usd", $RobotUsd,
    "--planner-mode", $PlannerMode,
    "--goal-x", $GoalX,
    "--goal-y", $GoalY,
    "--instruction", $Instruction,
    "--server-url", $ServerUrl,
    "--dual-server-url", $DualServerUrl
)

if (-not [string]::IsNullOrWhiteSpace($SceneUsd)) {
    $LaunchArgs += @("--env-url", $SceneUsd)
}

if ($ForceRuntimeCamera -notin @("0", "false", "False", "FALSE", "no", "No", "NO")) {
    $LaunchArgs += @("--force-runtime-camera")
}

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $IsaacPython @LaunchArgs @args
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
