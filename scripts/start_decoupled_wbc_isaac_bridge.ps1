param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacPythonExe = "",
  [string]$DecoupledPythonExe = "python",
  [string]$DecoupledWbcRoot = "",
  [string]$UsdPath = "",
  [string]$Namespace = "g1",
  [string]$RosDomainId = "0",
  [string]$RmwImplementation = "rmw_fastrtps_cpp",
  [string]$InternalStateTopic = "G1Env/isaac_state",
  [string]$InternalCommandTopic = "G1Env/isaac_joint_command",
  [double]$StatePublishHz = 100.0,
  [int]$ControlFrequency = 50,
  [switch]$StartTeleop = $false,
  [int]$TeleopFrequency = 20,
  [switch]$StartSonicServer = $false,
  [string]$SonicPythonExe = "",
  [string]$SonicModelDir = "",
  [switch]$IsaacGui = $false,
  [switch]$MockIsaac = $false,
  [switch]$SkipPatchSync = $false,
  [switch]$SkipControlLoop = $false
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

function Sync-DecoupledBridgeBundle {
  param(
    [string]$BundleRoot,
    [string]$TargetRoot
  )

  $fileMap = @(
    @{ source = "decoupled_wbc/control/main/constants.py"; target = "decoupled_wbc/control/main/constants.py" },
    @{ source = "decoupled_wbc/control/main/teleop/configs/configs.py"; target = "decoupled_wbc/control/main/teleop/configs/configs.py" },
    @{ source = "decoupled_wbc/control/main/teleop/run_g1_control_loop.py"; target = "decoupled_wbc/control/main/teleop/run_g1_control_loop.py" },
    @{ source = "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py"; target = "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py" },
    @{ source = "decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py"; target = "decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py" },
    @{ source = "decoupled_wbc/control/envs/g1/g1_body.py"; target = "decoupled_wbc/control/envs/g1/g1_body.py" },
    @{ source = "decoupled_wbc/control/envs/g1/g1_hand.py"; target = "decoupled_wbc/control/envs/g1/g1_hand.py" },
    @{ source = "decoupled_wbc/control/envs/g1/g1_env.py"; target = "decoupled_wbc/control/envs/g1/g1_env.py" },
    @{ source = "decoupled_wbc/control/envs/g1/sim/simulator_factory.py"; target = "decoupled_wbc/control/envs/g1/sim/simulator_factory.py" },
    @{ source = "decoupled_wbc/control/envs/g1/utils/isaac_ros_interface.py"; target = "decoupled_wbc/control/envs/g1/utils/isaac_ros_interface.py" },
    @{ source = "decoupled_wbc/control/utils/isaac_ros_adapter.py"; target = "decoupled_wbc/control/utils/isaac_ros_adapter.py" },
    @{ source = "tools/isaac/load_g1_usd_ros2.py"; target = "tools/isaac/load_g1_usd_ros2.py" }
  )

  foreach ($entry in $fileMap) {
    $src = Join-Path $BundleRoot $entry.source
    if (-not (Test-Path $src)) {
      throw "Bridge bundle source file not found: $src"
    }

    $dst = Join-Path $TargetRoot $entry.target
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path $dstDir)) {
      New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }

    Copy-Item -Path $src -Destination $dst -Force
    Write-Host "[start_decoupled_wbc_isaac_bridge] synced $($entry.source)"
  }
}

function Resolve-TopicFromNamespace {
  param(
    [string]$Ns,
    [string]$SuffixWithLeadingSlash,
    [string]$FallbackAbsolute
  )

  $trimmed = $Ns.Trim().Trim("/")
  if ([string]::IsNullOrWhiteSpace($trimmed)) {
    return $FallbackAbsolute
  }
  return "/$trimmed$SuffixWithLeadingSlash"
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundleRoot = Join-Path $root "apps/isaac_ros2_bridge_bundle"
if (-not (Test-Path $bundleRoot)) {
  throw "Bridge bundle not found: $bundleRoot"
}

if ([string]::IsNullOrWhiteSpace($DecoupledWbcRoot)) {
  $DecoupledWbcRoot = Join-Path $root "apps/decoupled_wbc_workspace"
}
if (-not [System.IO.Path]::IsPathRooted($DecoupledWbcRoot)) {
  $DecoupledWbcRoot = Join-Path $root $DecoupledWbcRoot
}
if (-not (Test-Path $DecoupledWbcRoot)) {
  throw "Decoupled WBC root not found: $DecoupledWbcRoot"
}
$DecoupledWbcRoot = (Resolve-Path $DecoupledWbcRoot).Path

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
$DecoupledPythonExe = Resolve-Executable -Candidate $DecoupledPythonExe -Label "DecoupledPythonExe"

if (-not $SkipPatchSync) {
  Sync-DecoupledBridgeBundle -BundleRoot $bundleRoot -TargetRoot $DecoupledWbcRoot
}

$adapterScript = Join-Path $DecoupledWbcRoot "decoupled_wbc/control/main/teleop/run_isaac_ros2_adapter.py"
$controlScript = Join-Path $DecoupledWbcRoot "decoupled_wbc/control/main/teleop/run_g1_control_loop.py"
$teleopScript = Join-Path $DecoupledWbcRoot "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py"

$requiredScripts = @($adapterScript)
if (-not $SkipControlLoop) {
  $requiredScripts += $controlScript
}
foreach ($required in $requiredScripts) {
  if (-not (Test-Path $required)) {
    throw "Required script missing after sync: $required"
  }
}
if ($StartTeleop -and -not (Test-Path $teleopScript)) {
  throw "Teleop script not found: $teleopScript"
}

$env:ISAAC_SIM_ROOT = $IsaacSimRoot
$env:ROS_DOMAIN_ID = $RosDomainId
$env:RMW_IMPLEMENTATION = $RmwImplementation
$env:PYTHONPATH = "$DecoupledWbcRoot;$root;$($env:PYTHONPATH)"

$jointStatesTopic = Resolve-TopicFromNamespace -Ns $Namespace -SuffixWithLeadingSlash "/joint_states" -FallbackAbsolute "/joint_states"
$imuTopic = Resolve-TopicFromNamespace -Ns $Namespace -SuffixWithLeadingSlash "/imu" -FallbackAbsolute "/imu"
$isaacCommandTopic = Resolve-TopicFromNamespace -Ns $Namespace -SuffixWithLeadingSlash "/cmd/joint_commands" -FallbackAbsolute "/cmd/joint_commands"

Write-Host "[start_decoupled_wbc_isaac_bridge] root=$root"
Write-Host "[start_decoupled_wbc_isaac_bridge] decoupled_root=$DecoupledWbcRoot"
Write-Host "[start_decoupled_wbc_isaac_bridge] usd=$UsdPath"
Write-Host "[start_decoupled_wbc_isaac_bridge] namespace=/$($Namespace.Trim('/'))"
Write-Host "[start_decoupled_wbc_isaac_bridge] topics: joint_states=$jointStatesTopic imu=$imuTopic tf=/tf clock=/clock cmd=$isaacCommandTopic"
Write-Host "[start_decoupled_wbc_isaac_bridge] internal_topics: state=$InternalStateTopic command=$InternalCommandTopic"

$procs = @()
try {
  $isaacArgs = @(
    "$root/apps/isaacsim_runner/run_headless.py",
    "--usd", "$UsdPath",
    "--namespace", "$Namespace",
    "--publish-imu",
    "--log-level", "INFO"
  )
  if ($IsaacGui) { $isaacArgs += "--gui" }
  if ($MockIsaac) { $isaacArgs += "--mock" }

  $isaacProc = Start-Process -FilePath $IsaacPythonExe -ArgumentList $isaacArgs -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += [pscustomobject]@{ Name = "isaac_sim"; Process = $isaacProc }
  Write-Host "[start_decoupled_wbc_isaac_bridge] isaac_sim pid=$($isaacProc.Id)"
  Start-Sleep -Seconds 3

  $adapterArgs = @(
    "$adapterScript",
    "--run-mode", "both",
    "--with-hands", "True",
    "--state-publish-hz", "$StatePublishHz",
    "--joint-states-topic", "$jointStatesTopic",
    "--imu-topic", "$imuTopic",
    "--tf-topic", "/tf",
    "--clock-topic", "/clock",
    "--internal-state-topic", "$InternalStateTopic",
    "--internal-command-topic", "$InternalCommandTopic",
    "--isaac-command-topic", "$isaacCommandTopic"
  )
  $adapterProc = Start-Process -FilePath $DecoupledPythonExe -ArgumentList $adapterArgs -WorkingDirectory $DecoupledWbcRoot -PassThru -NoNewWindow
  $procs += [pscustomobject]@{ Name = "isaac_ros2_adapter"; Process = $adapterProc }
  Write-Host "[start_decoupled_wbc_isaac_bridge] isaac_ros2_adapter pid=$($adapterProc.Id)"
  Start-Sleep -Seconds 2

  if (-not $SkipControlLoop) {
    $controlArgs = @(
      "$controlScript",
      "--interface", "sim",
      "--simulator", "isaacsim",
      "--control-frequency", "$ControlFrequency",
      "--isaac-internal-state-topic", "$InternalStateTopic",
      "--isaac-internal-command-topic", "$InternalCommandTopic"
    )
    $controlProc = Start-Process -FilePath $DecoupledPythonExe -ArgumentList $controlArgs -WorkingDirectory $DecoupledWbcRoot -PassThru -NoNewWindow
    $procs += [pscustomobject]@{ Name = "decoupled_wbc_control"; Process = $controlProc }
    Write-Host "[start_decoupled_wbc_isaac_bridge] decoupled_wbc_control pid=$($controlProc.Id)"
  } else {
    Write-Host "[start_decoupled_wbc_isaac_bridge] skipping decoupled_wbc_control (-SkipControlLoop)."
  }

  if ($StartTeleop) {
    $teleopArgs = @(
      "$teleopScript",
      "--interface", "sim",
      "--simulator", "isaacsim",
      "--teleop-frequency", "$TeleopFrequency"
    )
    $teleopProc = Start-Process -FilePath $DecoupledPythonExe -ArgumentList $teleopArgs -WorkingDirectory $DecoupledWbcRoot -PassThru -NoNewWindow
    $procs += [pscustomobject]@{ Name = "decoupled_wbc_teleop"; Process = $teleopProc }
    Write-Host "[start_decoupled_wbc_isaac_bridge] decoupled_wbc_teleop pid=$($teleopProc.Id)"
  }

  if ($StartSonicServer) {
    $sonicLauncher = Join-Path $root "scripts/start_sonic_server.ps1"
    if (-not (Test-Path $sonicLauncher)) {
      throw "SONIC launcher not found: $sonicLauncher"
    }

    $sonicPsArgs = @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", "$sonicLauncher"
    )
    if (-not [string]::IsNullOrWhiteSpace($SonicPythonExe)) {
      $sonicPsArgs += @("-PythonExe", "$SonicPythonExe")
    }
    if (-not [string]::IsNullOrWhiteSpace($SonicModelDir)) {
      $sonicPsArgs += @("-ModelDir", "$SonicModelDir")
    }

    $sonicProc = Start-Process -FilePath "powershell" -ArgumentList $sonicPsArgs -WorkingDirectory $root -PassThru -NoNewWindow
    $procs += [pscustomobject]@{ Name = "gear_sonic"; Process = $sonicProc }
    Write-Host "[start_decoupled_wbc_isaac_bridge] gear_sonic pid=$($sonicProc.Id)"
  }

  Write-Host "[start_decoupled_wbc_isaac_bridge] running. Press Ctrl+C to stop all child processes."
  while ($true) {
    Start-Sleep -Seconds 1
    $exited = @()
    foreach ($entry in $procs) {
      $proc = $entry.Process
      try {
        $proc.Refresh()
        if ($proc.HasExited) {
          $exited += $entry
        }
      } catch {
        $exited += $entry
      }
    }

    if ($exited.Count -gt 0) {
      foreach ($entry in $exited) {
        $code = "unknown"
        try { $code = [string]$entry.Process.ExitCode } catch { }
        Write-Warning "[start_decoupled_wbc_isaac_bridge] $($entry.Name) exited (code=$code). stopping remaining processes."
      }
      break
    }
  }
}
finally {
  Write-Host "[start_decoupled_wbc_isaac_bridge] stopping child processes..."
  foreach ($entry in $procs) {
    $proc = $entry.Process
    if ($null -ne $proc) {
      try {
        $proc.Refresh()
        if (-not $proc.HasExited) {
          Stop-Process -Id $proc.Id -Force
          Write-Host "[start_decoupled_wbc_isaac_bridge] stopped $($entry.Name) pid=$($proc.Id)"
        }
      } catch {
        Write-Host "[start_decoupled_wbc_isaac_bridge] stop failed for $($entry.Name): $($_.Exception.Message)"
      }
    }
  }
}
