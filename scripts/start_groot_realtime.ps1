param(
  [string]$PythonExe = "python",
  [string]$ConfigPath = "",
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$Command = "",
  [switch]$NoInteractive,
  [switch]$StartSonicServer = $true,
  [string]$SonicServerCommand = "",
  [string]$SonicServerCwd = "",
  [string]$SonicPythonExe = "",
  [string]$SonicHost = "127.0.0.1",
  [int]$SonicPort = 5556,
  [string]$SonicModelDir = ""
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"
$env:ISAAC_SIM_ROOT = $IsaacSimRoot

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

$resolvedConfig = $ConfigPath
if ([string]::IsNullOrWhiteSpace($resolvedConfig)) {
  $resolvedConfig = Join-Path $root "apps/agent_runtime/config.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($resolvedConfig)) {
  $resolvedConfig = Join-Path $root $resolvedConfig
}

if (-not (Test-Path $resolvedConfig)) {
  throw "Config file not found: $resolvedConfig"
}

$runnerExe = $PythonExe
$actionBackend = ""
$sonicEnabledInConfig = $true
try {
  $cfgObj = Get-Content $resolvedConfig -Raw | ConvertFrom-Json
  $actionBackend = [string]$cfgObj.manipulation.action_adapter.backend
  if ($null -ne $cfgObj.manipulation.sonic_server) {
    $sonicEnabledInConfig = [bool]$cfgObj.manipulation.sonic_server.enabled
    if ([string]::IsNullOrWhiteSpace($SonicHost)) {
      $SonicHost = [string]$cfgObj.manipulation.sonic_server.host
    }
    if ($SonicPort -le 0) {
      $SonicPort = [int]$cfgObj.manipulation.sonic_server.port
    }
  }
} catch {
  Write-Host "[start_groot_realtime] config parse warning: $($_.Exception.Message)"
}

if ($actionBackend -eq "ros2_topic") {
  $ros2BridgeRoot = Join-Path $IsaacSimRoot "exts\isaacsim.ros2.bridge\humble"
  $ros2Lib = Join-Path $ros2BridgeRoot "lib"
  $ros2Rclpy = Join-Path $ros2BridgeRoot "rclpy"
  if (Test-Path $ros2Lib) {
    if ([string]::IsNullOrWhiteSpace($env:ROS_DISTRO)) { $env:ROS_DISTRO = "humble" }
    if ([string]::IsNullOrWhiteSpace($env:RMW_IMPLEMENTATION)) { $env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp" }
    if ($env:PATH -notlike "*$ros2Lib*") { $env:PATH = "$env:PATH;$ros2Lib" }
    if (Test-Path $ros2Rclpy) {
      if ($env:PYTHONPATH -notlike "*$ros2Rclpy*") {
        $env:PYTHONPATH = "$ros2Rclpy;$($env:PYTHONPATH)"
      }
    }
    Write-Host "[start_groot_realtime] configured internal ROS2 bridge environment (humble)"
  }

  if (-not (Test-PythonImports -Exe $runnerExe -Imports "rclpy")) {
    $isaacPy = Join-Path $IsaacSimRoot "python.bat"
    if (Test-Path $isaacPy) {
      if (Test-PythonImports -Exe $isaacPy -Imports "rclpy, numpy, msgpack, zmq") {
        $runnerExe = $isaacPy
        Write-Warning "Switching realtime GR00T runner to Isaac Sim python: $runnerExe"
      } else {
        Write-Warning "Cannot import one or more modules (rclpy/numpy/msgpack/zmq). Install deps with:"
        Write-Warning "  C:\isaac-sim\python.bat -m pip install numpy pyzmq msgpack"
      }
    }
  }
}

Write-Host "[start_groot_realtime] root=$root"
Write-Host "[start_groot_realtime] config=$resolvedConfig"
Write-Host "[start_groot_realtime] python=$runnerExe"
Write-Host "[start_groot_realtime] StartSonicServer=$StartSonicServer"

$args = @(
  "$root/apps/agent_runtime/realtime_groot_cli.py",
  "--config", "$resolvedConfig"
)
if (-not [string]::IsNullOrWhiteSpace($Command)) {
  $args += "--command"
  $args += "$Command"
}
if ($NoInteractive) { $args += "--no-interactive" }

$sonicProc = $null
try {
  if ($StartSonicServer -and $sonicEnabledInConfig) {
    if ([string]::IsNullOrWhiteSpace($SonicServerCommand)) {
      $resolvedSonicModelDir = if ([string]::IsNullOrWhiteSpace($SonicModelDir)) {
        Join-Path $root "gear_sonic_deploy"
      } elseif ([System.IO.Path]::IsPathRooted($SonicModelDir)) {
        $SonicModelDir
      } else {
        Join-Path $root $SonicModelDir
      }

      if (-not (Test-Path $resolvedSonicModelDir)) {
        throw "SONIC model dir not found: $resolvedSonicModelDir"
      }
      $resolvedSonicModelDir = (Resolve-Path $resolvedSonicModelDir).Path

      $encoderPath = Join-Path $resolvedSonicModelDir "model_encoder.onnx"
      $decoderPath = Join-Path $resolvedSonicModelDir "model_decoder.onnx"
      $plannerPath = Join-Path $resolvedSonicModelDir "planner_sonic.onnx"
      foreach ($p in @($encoderPath, $decoderPath, $plannerPath)) {
        if (-not (Test-Path $p)) {
          throw "SONIC model file not found: $p"
        }
      }

      $sonicArgs = @(
        "$root/sonic_policy_server.py",
        "--encoder", "$encoderPath",
        "--decoder", "$decoderPath",
        "--planner", "$plannerPath",
        "--host", "$SonicHost",
        "--port", "$SonicPort"
      )
      $sonicExe = $SonicPythonExe
      if ([string]::IsNullOrWhiteSpace($sonicExe)) {
        $sonicExe = $runnerExe
      }

      if (-not (Test-PythonImports -Exe $sonicExe -Imports "onnxruntime, numpy, msgpack, zmq")) {
        if (-not [string]::IsNullOrWhiteSpace($SonicPythonExe)) {
          throw "SONIC python import check failed in '$SonicPythonExe'. Required: onnxruntime, numpy, msgpack, zmq."
        }

        $fallbackSonicExe = ""
        foreach ($cand in @("python", "py")) {
          if (Test-PythonImports -Exe $cand -Imports "onnxruntime, numpy, msgpack, zmq") {
            $fallbackSonicExe = $cand
            break
          }
        }

        if ([string]::IsNullOrWhiteSpace($fallbackSonicExe)) {
          throw "No usable python found for SONIC server. Provide -SonicPythonExe with an environment that imports onnxruntime, numpy, msgpack, zmq."
        }

        Write-Warning "Switching SONIC server python to '$fallbackSonicExe' (onnxruntime import check passed)."
        $sonicExe = $fallbackSonicExe
      }

      $sonicProc = Start-Process -FilePath $sonicExe -ArgumentList $sonicArgs -WorkingDirectory $root -PassThru -NoNewWindow
    } else {
      $sonicCwd = $root
      if (-not [string]::IsNullOrWhiteSpace($SonicServerCwd)) {
        if (-not (Test-Path $SonicServerCwd)) {
          throw "Invalid -SonicServerCwd path: $SonicServerCwd"
        }
        $sonicCwd = $SonicServerCwd
      }
      $sonicProc = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $SonicServerCommand) -WorkingDirectory $sonicCwd -PassThru -NoNewWindow
    }
    if ([string]::IsNullOrWhiteSpace($SonicServerCommand)) {
      Write-Host "[start_groot_realtime] sonic_policy_server pid=$($sonicProc.Id) host=$SonicHost port=$SonicPort python=$sonicExe"
    } else {
      Write-Host "[start_groot_realtime] sonic_policy_server pid=$($sonicProc.Id) host=$SonicHost port=$SonicPort"
    }
    Start-Sleep -Seconds 2
  } elseif ($StartSonicServer -and -not $sonicEnabledInConfig) {
    Write-Host "[start_groot_realtime] sonic_server.enabled=false in config; skipping SONIC server startup."
  }

  & $runnerExe @args
}
finally {
  if ($null -ne $sonicProc) {
    try {
      $sonicProc.Refresh()
      if (-not $sonicProc.HasExited) {
        Stop-Process -Id $sonicProc.Id -Force
        Write-Host "[start_groot_realtime] stopped sonic_policy_server pid=$($sonicProc.Id)"
      }
    } catch {
      Write-Host "[start_groot_realtime] stop failed for sonic pid=$($sonicProc.Id): $($_.Exception.Message)"
    }
  }
}
