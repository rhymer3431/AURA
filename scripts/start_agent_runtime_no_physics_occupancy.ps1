param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacPythonExe = "",
  [string]$G1UsdPath = "",
  [string]$StageUsdPath = "",
  [string]$MapDir = "",
  [string]$MapName = "g1_no_physics",
  [double]$Resolution = 0.10,
  [double]$MapWidthM = 20.0,
  [double]$MapHeightM = 20.0,
  [double]$OriginX = -10.0,
  [double]$OriginY = -10.0,
  [string]$Namespace = "g1",
  [double]$MoveDurationS = 1.5,
  [string]$RobotPrimPath = "/World/Robots/G1",
  [string]$StartGoalCell = "",
  [string]$ObstaclesJson = "",
  [switch]$Headless = $false,
  [switch]$SkipStageBuild = $false,
  [switch]$DisableRos2Goals = $false,
  [switch]$DisableStdinGoals = $false,
  [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

function Resolve-Executable {
  param(
    [string]$Candidate,
    [string]$Label
  )

  if ([string]::IsNullOrWhiteSpace($Candidate)) {
    throw "Missing executable for $Label"
  }

  if ([System.IO.Path]::IsPathRooted($Candidate)) {
    if (-not (Test-Path $Candidate)) {
      throw "Invalid $Label path: $Candidate"
    }
    return (Resolve-Path $Candidate).Path
  }

  $cmd = Get-Command $Candidate -ErrorAction SilentlyContinue
  if ($null -eq $cmd) {
    throw "$Label command not found in PATH: $Candidate"
  }
  return $Candidate
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $root "apps/isaacsim_runner/agent_runtime/build_no_physics_occupancy_stage.py"
$controllerScript = Join-Path $root "apps/isaacsim_runner/agent_runtime/run_no_physics_occupancy_controller.py"

if (-not (Test-Path $buildScript)) {
  throw "Stage builder script not found: $buildScript"
}
if (-not (Test-Path $controllerScript)) {
  throw "Controller script not found: $controllerScript"
}

if ([string]::IsNullOrWhiteSpace($G1UsdPath)) {
  $G1UsdPath = Join-Path $root "apps/isaac_ros2_bridge_bundle/robot_model/model_data/g1/g1_29dof_with_hand/g1_29dof_with_hand.usd"
} elseif (-not [System.IO.Path]::IsPathRooted($G1UsdPath)) {
  $G1UsdPath = Join-Path $root $G1UsdPath
}
if (-not (Test-Path $G1UsdPath)) {
  throw "G1 USD path not found: $G1UsdPath"
}
$G1UsdPath = (Resolve-Path $G1UsdPath).Path

if ([string]::IsNullOrWhiteSpace($StageUsdPath)) {
  $StageUsdPath = Join-Path $root "tmp/agent_runtime/g1_no_physics_stage.usda"
} elseif (-not [System.IO.Path]::IsPathRooted($StageUsdPath)) {
  $StageUsdPath = Join-Path $root $StageUsdPath
}
$stageDir = Split-Path -Parent $StageUsdPath
if (-not (Test-Path $stageDir)) {
  New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
}

if ([string]::IsNullOrWhiteSpace($MapDir)) {
  $MapDir = Join-Path $root "tmp/agent_runtime/maps"
} elseif (-not [System.IO.Path]::IsPathRooted($MapDir)) {
  $MapDir = Join-Path $root $MapDir
}
if (-not (Test-Path $MapDir)) {
  New-Item -ItemType Directory -Path $MapDir -Force | Out-Null
}

$mapYamlPath = Join-Path $MapDir "$MapName.yaml"

if ([string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
  $candidate = Join-Path $IsaacSimRoot "python.bat"
  if (-not (Test-Path $candidate)) {
    throw "Isaac Sim python launcher not found: $candidate. Pass -IsaacPythonExe explicitly."
  }
  $IsaacPythonExe = $candidate
}
$IsaacPythonExe = Resolve-Executable -Candidate $IsaacPythonExe -Label "IsaacPythonExe"

$env:ISAAC_SIM_ROOT = $IsaacSimRoot
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"

Write-Host "[start_agent_runtime_no_physics_occupancy] root=$root"
Write-Host "[start_agent_runtime_no_physics_occupancy] python=$IsaacPythonExe"
Write-Host "[start_agent_runtime_no_physics_occupancy] g1_usd=$G1UsdPath"
Write-Host "[start_agent_runtime_no_physics_occupancy] stage=$StageUsdPath"
Write-Host "[start_agent_runtime_no_physics_occupancy] map_yaml=$mapYamlPath"
Write-Host "[start_agent_runtime_no_physics_occupancy] namespace=/$($Namespace.Trim('/'))"

if (-not $SkipStageBuild) {
  $buildArgs = @(
    "$buildScript",
    "--g1-usd", "$G1UsdPath",
    "--stage-out", "$StageUsdPath",
    "--map-dir", "$MapDir",
    "--map-name", "$MapName",
    "--resolution", "$Resolution",
    "--map-width-m", "$MapWidthM",
    "--map-height-m", "$MapHeightM",
    "--origin-x", "$OriginX",
    "--origin-y", "$OriginY",
    "--robot-prim-path", "$RobotPrimPath",
    "--log-level", "$LogLevel"
  )
  if (-not [string]::IsNullOrWhiteSpace($ObstaclesJson)) {
    if (-not [System.IO.Path]::IsPathRooted($ObstaclesJson)) {
      $ObstaclesJson = Join-Path $root $ObstaclesJson
    }
    $buildArgs += @("--obstacles-json", "$ObstaclesJson")
  }
  if ($Headless) {
    $buildArgs += "--headless"
  }

  Write-Host "[start_agent_runtime_no_physics_occupancy] building no-physics stage + occupancy map..."
  & $IsaacPythonExe @buildArgs
  $buildExit = $LASTEXITCODE
  if ($buildExit -ne 0) {
    throw "Stage build failed with exit code $buildExit"
  }
} else {
  Write-Host "[start_agent_runtime_no_physics_occupancy] skipping stage build (-SkipStageBuild)."
}

if (-not (Test-Path $StageUsdPath)) {
  throw "Stage USD missing: $StageUsdPath"
}
if (-not (Test-Path $mapYamlPath)) {
  throw "Occupancy map YAML missing: $mapYamlPath"
}

$runArgs = @(
  "$controllerScript",
  "--stage-usd", "$StageUsdPath",
  "--occupancy-map-yaml", "$mapYamlPath",
  "--robot-prim-path", "$RobotPrimPath",
  "--namespace", "$Namespace",
  "--move-duration-s", "$MoveDurationS",
  "--log-level", "$LogLevel"
)
if ($Headless) {
  $runArgs += "--headless"
}
if ($DisableRos2Goals) {
  $runArgs += "--disable-ros2-goals"
}
if (-not $DisableStdinGoals) {
  $runArgs += "--enable-stdin-goals"
}
if (-not [string]::IsNullOrWhiteSpace($StartGoalCell)) {
  $runArgs += @("--start-goal-cell", "$StartGoalCell")
}

Write-Host "[start_agent_runtime_no_physics_occupancy] starting occupancy controller..."
Write-Host "[start_agent_runtime_no_physics_occupancy] ROS2 goal topic: /$($Namespace.Trim('/'))/cmd/occupancy_goal (Point.x=mx, y=my, z=yaw_deg)"
Write-Host "[start_agent_runtime_no_physics_occupancy] stdin format: <mx> <my> [yaw_deg]"

& $IsaacPythonExe @runArgs
$runExit = $LASTEXITCODE
if ($runExit -ne 0) {
  throw "Occupancy controller exited with code $runExit"
}

