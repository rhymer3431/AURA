param(
  [string]$PythonExe = "C:\isaac-sim\python.bat",
  [string]$ConfigPath = "",
  [string]$Command = "",
  [switch]$Keyboard,
  [switch]$NoInteractive
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"
$env:ISAAC_SIM_ROOT = if ([string]::IsNullOrWhiteSpace($env:ISAAC_SIM_ROOT)) { "C:\isaac-sim" } else { $env:ISAAC_SIM_ROOT }

function Test-PythonImports {
  param(
    [string]$Exe,
    [string]$Imports
  )
  try {
    & $Exe -c "import $Imports" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

if (-not (Test-Path $PythonExe)) {
  Write-Warning "Python exe not found: $PythonExe. Falling back to 'python'."
  $PythonExe = "python"
}

$isaacRoot = $env:ISAAC_SIM_ROOT
$ros2BridgeRoot = Join-Path $isaacRoot "exts\isaacsim.ros2.bridge\humble"
$ros2Lib = Join-Path $ros2BridgeRoot "lib"
$ros2Rclpy = Join-Path $ros2BridgeRoot "rclpy"
if (Test-Path $ros2Lib) {
  if ([string]::IsNullOrWhiteSpace($env:ROS_DISTRO)) { $env:ROS_DISTRO = "humble" }
  if ([string]::IsNullOrWhiteSpace($env:RMW_IMPLEMENTATION)) { $env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp" }

  if ($env:PATH -notlike "*$ros2Lib*") {
    $env:PATH = "$env:PATH;$ros2Lib"
  }
  if (Test-Path $ros2Rclpy) {
    if ($env:PYTHONPATH -notlike "*$ros2Rclpy*") {
      $env:PYTHONPATH = "$ros2Rclpy;$($env:PYTHONPATH)"
    }
  }
  Write-Host "[start_sonic_only] configured internal ROS2 bridge environment (humble)"
}

if (-not (Test-PythonImports -Exe $PythonExe -Imports "rclpy, numpy, msgpack, zmq")) {
  $isaacPy = Join-Path $isaacRoot "python.bat"
  if ((Test-Path $isaacPy) -and (Test-PythonImports -Exe $isaacPy -Imports "rclpy, numpy, msgpack, zmq")) {
    Write-Warning "Current python cannot import required runtime modules. Switching to Isaac Python: $isaacPy"
    $PythonExe = $isaacPy
  } else {
    throw "Python '$PythonExe' cannot import required modules (rclpy, numpy, msgpack, zmq)."
  }
}

$resolvedConfig = $ConfigPath
if ([string]::IsNullOrWhiteSpace($resolvedConfig)) {
  $resolvedConfig = Join-Path $root "apps/agent_runtime/config.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($resolvedConfig)) {
  $resolvedConfig = Join-Path $root $resolvedConfig
}

if (-not (Test-Path $resolvedConfig)) {
  throw "Config file not found: $resolvedConfig"
}

Write-Host "[start_sonic_only] root=$root"
Write-Host "[start_sonic_only] python=$PythonExe"
Write-Host "[start_sonic_only] config=$resolvedConfig"
Write-Host "[start_sonic_only] keyboard=$Keyboard"

$args = @(
  "$root/apps/agent_runtime/sonic_only_cli.py",
  "--config", "$resolvedConfig"
)

if (-not [string]::IsNullOrWhiteSpace($Command)) {
  $args += "--command"
  $args += "$Command"
}
if ($Keyboard) { $args += "--keyboard" }
if ($NoInteractive) { $args += "--no-interactive" }

& $PythonExe @args
