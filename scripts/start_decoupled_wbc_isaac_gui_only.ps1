param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacPythonExe = "",
  [string]$UsdPath = "",
  [string]$Namespace = "g1",
  [string]$LogLevel = "INFO",
  [switch]$MockIsaac = $false,
  [switch]$PublishCompressedColor = $false,
  [switch]$EnableCameraBridgeInGui = $false,
  [switch]$EnableNavigateBridge = $false
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
$runnerScript = Join-Path $root "apps/isaacsim_runner/run_headless.py"
if (-not (Test-Path $runnerScript)) {
  throw "Isaac runner script not found: $runnerScript"
}

if ([string]::IsNullOrWhiteSpace($UsdPath)) {
  $UsdPath = Join-Path $root "apps/isaac_ros2_bridge_bundle/robot_model/model_data/g1/g1_29dof_with_hand/g1_29dof_with_hand.usd"
} elseif (-not [System.IO.Path]::IsPathRooted($UsdPath)) {
  $UsdPath = Join-Path $root $UsdPath
}
if (-not (Test-Path $UsdPath)) {
  throw "USD path not found: $UsdPath"
}
$UsdPath = (Resolve-Path $UsdPath).Path

if ([string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
  $candidate = Join-Path $IsaacSimRoot "python.bat"
  if (Test-Path $candidate) {
    $IsaacPythonExe = $candidate
  } elseif ($MockIsaac) {
    $IsaacPythonExe = "python"
  } else {
    throw "Isaac Sim python launcher not found: $candidate. Pass -IsaacPythonExe explicitly."
  }
}
$IsaacPythonExe = Resolve-Executable -Candidate $IsaacPythonExe -Label "IsaacPythonExe"

$env:ISAAC_SIM_ROOT = $IsaacSimRoot

$isaacArgs = @(
  "$runnerScript",
  "--usd", "$UsdPath",
  "--namespace", "$Namespace",
  "--publish-imu",
  "--log-level", "$LogLevel",
  "--gui"
)
if ($PublishCompressedColor) { $isaacArgs += "--publish-compressed-color" }
if ($EnableCameraBridgeInGui) { $isaacArgs += "--enable-camera-bridge-in-gui" }
if ($EnableNavigateBridge) { $isaacArgs += "--enable-navigate-bridge" }
if ($MockIsaac) { $isaacArgs += "--mock" }

Write-Host "[start_decoupled_wbc_isaac_gui_only] root=$root"
Write-Host "[start_decoupled_wbc_isaac_gui_only] ISAAC_SIM_ROOT=$IsaacSimRoot"
Write-Host "[start_decoupled_wbc_isaac_gui_only] runner=$runnerScript"
Write-Host "[start_decoupled_wbc_isaac_gui_only] usd=$UsdPath"
Write-Host "[start_decoupled_wbc_isaac_gui_only] namespace=/$($Namespace.Trim('/'))"
Write-Host "[start_decoupled_wbc_isaac_gui_only] exe=$IsaacPythonExe"
Write-Host "[start_decoupled_wbc_isaac_gui_only] launching Isaac Sim GUI..."

& $IsaacPythonExe @isaacArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  throw "Isaac Sim runner exited with code $exitCode"
}
