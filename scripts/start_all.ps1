param(
  [string]$PythonExe = "python",
  [string]$AgentPythonExe = "C:\isaac-sim\python.bat",
  [string]$Command = "Pick up the apple and return",
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacLabRoot = "C:\Users\mango\project\isaac-lab",
  [string]$IsaacPythonExe = "",
  [switch]$StartGrootServer = $false,
  [string]$GrootServerCommand = "",
  [string]$GrootServerCwd = "",
  [switch]$StartSonicServer = $true,
  [string]$SonicServerCommand = "",
  [string]$SonicServerCwd = "",
  [string]$SonicHost = "127.0.0.1",
  [int]$SonicPort = 5556,
  [string]$SonicModelDir = "",
  [double]$SonicActionScaleMultiplier = 0.10,
  [switch]$MockIsaac = $false,
  [switch]$IsaacGui = $false,
  [switch]$MockPlanner = $false,
  [switch]$NoInteractive = $false,
  [switch]$PublishCompressedColor = $false
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"
$env:ISAAC_SIM_ROOT = $IsaacSimRoot
$env:ISAACLAB_ROOT = $IsaacLabRoot

Write-Host "[start_all] root=$root"
Write-Host "[start_all] command=$Command"
Write-Host "[start_all] ISAAC_SIM_ROOT=$IsaacSimRoot"
Write-Host "[start_all] ISAACLAB_ROOT=$IsaacLabRoot"
Write-Host "[start_all] PythonExe=$PythonExe"
if (-not [string]::IsNullOrWhiteSpace($AgentPythonExe)) {
  Write-Host "[start_all] AgentPythonExe=$AgentPythonExe"
}
Write-Host "[start_all] StartGrootServer=$StartGrootServer"
Write-Host "[start_all] StartSonicServer=$StartSonicServer"
Write-Host "[start_all] SonicActionScaleMultiplier=$SonicActionScaleMultiplier"
Write-Host "[start_all] IsaacGui=$IsaacGui"
Write-Host "[start_all] MockIsaac=$MockIsaac"
Write-Host "[start_all] MockPlanner=$MockPlanner"
if ($MockIsaac) {
  Write-Warning "MockIsaac=true: no real Isaac Sim physics/articulation control. Robot motion will not execute."
}
if ($MockPlanner) {
  Write-Warning "MockPlanner=true: planner output is stubbed and not model-based."
}

$cfgPath = Join-Path $root "apps/agent_runtime/config.yaml"
$cfgObj = $null
$actionBackend = ""
$sonicEnabledInConfig = $true
$cfgSonicHost = ""
$cfgSonicPort = 0
$locomotionBackend = "sonic_server"
if (Test-Path $cfgPath) {
  try {
    $cfgObj = Get-Content $cfgPath -Raw | ConvertFrom-Json
    $actionBackend = [string]$cfgObj.manipulation.action_adapter.backend
    $cfgLocoBackend = [string]$cfgObj.manipulation.locomotion_backend
    if (-not [string]::IsNullOrWhiteSpace($cfgLocoBackend)) {
      $locomotionBackend = $cfgLocoBackend.Trim().ToLowerInvariant()
    }
    if ($null -ne $cfgObj.manipulation.sonic_server) {
      $sonicEnabledInConfig = [bool]$cfgObj.manipulation.sonic_server.enabled
      $cfgSonicHost = [string]$cfgObj.manipulation.sonic_server.host
      $cfgSonicPort = [int]$cfgObj.manipulation.sonic_server.port
    }
  } catch {
    Write-Host "[start_all] config parse warning: $($_.Exception.Message)"
  }
}

if ([string]::IsNullOrWhiteSpace($SonicHost) -and -not [string]::IsNullOrWhiteSpace($cfgSonicHost)) {
  $SonicHost = $cfgSonicHost
}
if ($SonicPort -le 0 -and $cfgSonicPort -gt 0) {
  $SonicPort = $cfgSonicPort
}

$isaacRunnerExe = $PythonExe
$plannerPythonExe = $PythonExe
$agentRunnerExe = if ([string]::IsNullOrWhiteSpace($AgentPythonExe)) { $PythonExe } else { $AgentPythonExe }
if (-not $MockIsaac) {
  if ([string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
    $candidate = Join-Path $IsaacSimRoot "python.bat"
    if (Test-Path $candidate) {
      $isaacRunnerExe = $candidate
    } else {
      throw "Isaac Sim python launcher not found: $candidate. Pass -IsaacPythonExe explicitly."
    }
  } else {
    if (-not (Test-Path $IsaacPythonExe)) {
      throw "Invalid -IsaacPythonExe path: $IsaacPythonExe"
    }
    $isaacRunnerExe = $IsaacPythonExe
  }
}

if ($actionBackend -eq "ros2_topic") {
  $ros2BridgeRoot = Join-Path $IsaacSimRoot "exts\isaacsim.ros2.bridge\humble"
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
        if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
          $env:PYTHONPATH = $ros2Rclpy
        } else {
          $env:PYTHONPATH = "$ros2Rclpy;$($env:PYTHONPATH)"
        }
      }
    }
    Write-Host "[start_all] configured internal ROS2 bridge environment (humble)"
  }

  $agentHasRclpy = $false
  & $agentRunnerExe -c "import rclpy" 2>$null
  if ($LASTEXITCODE -eq 0) {
    $agentHasRclpy = $true
  }

  if (-not $agentHasRclpy -and -not $MockIsaac) {
    & $isaacRunnerExe -c "import rclpy, numpy, msgpack, zmq" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $agentRunnerExe = $isaacRunnerExe
      Write-Warning "action_adapter.backend=ros2_topic and rclpy is unavailable in PythonExe. Switching agent_runtime to Isaac Sim python: $agentRunnerExe"
      $agentHasRclpy = $true
    } else {
      Write-Warning "Isaac Sim python cannot import one or more required modules (rclpy/numpy/msgpack/zmq). Install runtime deps with:"
      Write-Warning "  C:\isaac-sim\python.bat -m pip install numpy pyzmq msgpack"
    }
  }

  if (-not $agentHasRclpy) {
    Write-Warning "action_adapter.backend=ros2_topic but rclpy is unavailable in agent runtime python ($agentRunnerExe). Commands will not be published to ROS2 topics."
  }
}

Write-Host "[start_all] planner_python=$plannerPythonExe"
Write-Host "[start_all] agent_python=$agentRunnerExe"
Write-Host "[start_all] locomotion_backend=$locomotionBackend"

$procs = @()
try {
  if ($StartGrootServer) {
    if ([string]::IsNullOrWhiteSpace($GrootServerCommand)) {
      throw "When -StartGrootServer is set, -GrootServerCommand must be provided."
    }
    $grootCwd = $root
    if (-not [string]::IsNullOrWhiteSpace($GrootServerCwd)) {
      if (-not (Test-Path $GrootServerCwd)) {
        throw "Invalid -GrootServerCwd path: $GrootServerCwd"
      }
      $grootCwd = $GrootServerCwd
    }
    $groot = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $GrootServerCommand) -WorkingDirectory $grootCwd -PassThru -NoNewWindow
    $procs += $groot
    Write-Host "[start_all] gr00t_policy_server pid=$($groot.Id)"
    Start-Sleep -Seconds 3
  }

  $forceLegacySonic = -not [string]::IsNullOrWhiteSpace($SonicServerCommand)
  if ($StartSonicServer -and $sonicEnabledInConfig -and ($locomotionBackend -ne "direct_policy" -or $forceLegacySonic)) {
    if ([string]::IsNullOrWhiteSpace($SonicServerCommand)) {
      $resolvedSonicModelDir = if ([string]::IsNullOrWhiteSpace($SonicModelDir)) {
        Join-Path $root "apps/gear_sonic_deploy"
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
        "--port", "$SonicPort",
        "--action-scale-multiplier", "$SonicActionScaleMultiplier"
      )
      $sonic = Start-Process -FilePath $plannerPythonExe -ArgumentList $sonicArgs -WorkingDirectory $root -PassThru -NoNewWindow
    } else {
      $sonicCwd = $root
      if (-not [string]::IsNullOrWhiteSpace($SonicServerCwd)) {
        if (-not (Test-Path $SonicServerCwd)) {
          throw "Invalid -SonicServerCwd path: $SonicServerCwd"
        }
        $sonicCwd = $SonicServerCwd
      }
      $sonic = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $SonicServerCommand) -WorkingDirectory $sonicCwd -PassThru -NoNewWindow
    }

    $procs += $sonic
    Write-Host "[start_all] sonic_policy_server pid=$($sonic.Id) host=$SonicHost port=$SonicPort"
    Start-Sleep -Seconds 2
  } elseif ($StartSonicServer -and -not $sonicEnabledInConfig) {
    Write-Host "[start_all] sonic_server.enabled=false in config; skipping SONIC server startup."
  } elseif ($StartSonicServer -and $locomotionBackend -eq "direct_policy") {
    Write-Host "[start_all] locomotion_backend=direct_policy; skipping legacy SONIC server startup. (set -SonicServerCommand to force legacy start)"
  }

  $plannerArgs = @("-m", "apps.services.planner_server.server", "--host", "127.0.0.1", "--port", "8088")
  if ($MockPlanner) { $plannerArgs += "--mock" }
  $planner = Start-Process -FilePath $plannerPythonExe -ArgumentList $plannerArgs -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += $planner
  Write-Host "[start_all] planner_server pid=$($planner.Id)"
  Start-Sleep -Seconds 2

  $isaacArgs = @(
    "$root/apps/isaacsim_runner/isaac_runner.py",
    "--usd", "$root/g1/g1_d455.usd",
    "--namespace", "g1",
    "--log-level", "INFO"
  )
  if ($PublishCompressedColor) { $isaacArgs += "--publish-compressed-color" }
  if ($IsaacGui) { $isaacArgs += "--gui" }
  if ($MockIsaac) { $isaacArgs += "--mock" }
  $isaac = Start-Process -FilePath $isaacRunnerExe -ArgumentList $isaacArgs -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += $isaac
  Write-Host "[start_all] isaacsim_runner pid=$($isaac.Id) exe=$isaacRunnerExe"
  Start-Sleep -Seconds 2

  $agentArgs = @(
    "$root/apps/agent_runtime/main.py",
    "--config", "$root/apps/agent_runtime/config.yaml",
    "--command", "$Command"
  )
  if ($NoInteractive) { $agentArgs += "--no-interactive" }
  Write-Host "[start_all] launching agent_runtime..."
  & $agentRunnerExe @agentArgs
}
finally {
  Write-Host "[start_all] stopping child processes..."
  foreach ($proc in $procs) {
    if ($null -ne $proc) {
      try {
        $proc.Refresh()
        if (-not $proc.HasExited) {
          Stop-Process -Id $proc.Id -Force
          Write-Host "[start_all] stopped pid=$($proc.Id)"
        }
      } catch {
        Write-Host "[start_all] stop failed for pid=$($proc.Id): $($_.Exception.Message)"
      }
    }
  }
}
