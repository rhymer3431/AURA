$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "runtime.g1_bridge"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultIsaacPython = "C:\isaac-sim\python.bat"
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { $DefaultIsaacPython }

$PolicyCandidates = @(
    (Join-Path $RepoDir "artifacts\models\policy.onnx"),
    (Join-Path $RepoDir "policy.onnx")
)
$PolicyPath = $PolicyCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $PolicyPath) {
    $PolicyPath = $PolicyCandidates[0]
}
$RobotUsd = Join-Path $RepoDir "src\locomotion\g1\g1_d455.usd"
$PlannerMode = if ($env:G1_POINTGOAL_PLANNER_MODE) { $env:G1_POINTGOAL_PLANNER_MODE } else { "dual" }
$GoalX = if ($env:G1_POINTGOAL_GOAL_X) { $env:G1_POINTGOAL_GOAL_X } else { "2.0" }
$GoalY = if ($env:G1_POINTGOAL_GOAL_Y) { $env:G1_POINTGOAL_GOAL_Y } else { "0.0" }
$Instruction = if ($env:G1_POINTGOAL_INSTRUCTION) {
    $env:G1_POINTGOAL_INSTRUCTION
} else {
    "Navigate safely to the target and stop when complete."
}
$ServerUrl = if ($env:G1_POINTGOAL_SERVER_URL) { $env:G1_POINTGOAL_SERVER_URL } else { "http://127.0.0.1:8888" }
$DualServerUrl = if ($env:G1_POINTGOAL_DUAL_SERVER_URL) { $env:G1_POINTGOAL_DUAL_SERVER_URL } else { "http://127.0.0.1:8890" }
$Headless = if ($env:G1_POINTGOAL_HEADLESS) { $env:G1_POINTGOAL_HEADLESS } else { "1" }
$ForceRuntimeCamera = if ($env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA) { $env:G1_POINTGOAL_FORCE_RUNTIME_CAMERA } else { "1" }

$SceneDir = "C:\Users\mango\project\isaac\datasets\InteriorAgent\kujiale_0004"
$PreferredSceneUsd = Join-Path $SceneDir "kujiale_0004_navila_sanitized.usda"
$FallbackSceneUsd = Join-Path $SceneDir "kujiale_0004.usda"

$SceneUsd = if ($env:G1_POINTGOAL_SCENE_USD) {
    $env:G1_POINTGOAL_SCENE_USD
} elseif (Test-Path -LiteralPath $PreferredSceneUsd) {
    $PreferredSceneUsd
} else {
    $FallbackSceneUsd
}

if (-not (Test-Path -LiteralPath $IsaacPython)) {
    Write-Host "[G1 PointGoal Kujiale] Isaac python launcher not found: `"$IsaacPython`""
    Write-Host "[G1 PointGoal Kujiale] Set ISAAC_SIM_PYTHON to your python.bat path."
    exit 1
}

if (-not (Test-Path -LiteralPath $SrcDir)) {
    Write-Host "[G1 PointGoal Kujiale] src directory not found: `"$SrcDir`""
    exit 1
}

if (-not (Test-Path -LiteralPath $PolicyPath)) {
    Write-Host "[G1 PointGoal Kujiale] ONNX policy not found: `"$PolicyPath`""
    exit 1
}

if (-not (Test-Path -LiteralPath $RobotUsd)) {
    Write-Host "[G1 PointGoal Kujiale] G1 USD not found: `"$RobotUsd`""
    exit 1
}

if (-not (Test-Path -LiteralPath $SceneUsd)) {
    Write-Host "[G1 PointGoal Kujiale] Scene USD not found: `"$SceneUsd`""
    Write-Host "[G1 PointGoal Kujiale] Set G1_POINTGOAL_SCENE_USD to override the default scene file."
    exit 1
}

Write-Host "[G1 PointGoal Kujiale] Starting"
Write-Host "[G1 PointGoal Kujiale] python=`"$IsaacPython`""
Write-Host "[G1 PointGoal Kujiale] module=`"$EntryModule`""
Write-Host "[G1 PointGoal Kujiale] scene-usd=`"$SceneUsd`""
Write-Host "[G1 PointGoal Kujiale] policy=`"$PolicyPath`""
Write-Host "[G1 PointGoal Kujiale] robot-usd=`"$RobotUsd`""
Write-Host "[G1 PointGoal Kujiale] default planner-mode=$PlannerMode"
Write-Host "[G1 PointGoal Kujiale] default goal=($GoalX, $GoalY)"
Write-Host "[G1 PointGoal Kujiale] default instruction=`"$Instruction`""
Write-Host "[G1 PointGoal Kujiale] default server-url=$ServerUrl"
Write-Host "[G1 PointGoal Kujiale] default dual-server-url=$DualServerUrl"
Write-Host "[G1 PointGoal Kujiale] default headless=$Headless"
Write-Host "[G1 PointGoal Kujiale] default force-runtime-camera=$ForceRuntimeCamera"
Write-Host "[G1 PointGoal Kujiale] mode flag: --planner-mode pointgoal|dual"

$LaunchArgs = @(
    "-m", $EntryModule,
    "--scene-usd", $SceneUsd,
    "--policy", $PolicyPath,
    "--robot_usd", $RobotUsd,
    "--planner-mode", $PlannerMode,
    "--goal-x", $GoalX,
    "--goal-y", $GoalY,
    "--instruction", $Instruction,
    "--server-url", $ServerUrl,
    "--dual-server-url", $DualServerUrl
)
if ($Headless -notin @("0", "false", "False", "FALSE", "no", "No", "NO")) {
    $LaunchArgs += @("--headless")
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
