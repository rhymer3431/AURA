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
  [switch]$IsaacGui = $false,
  [switch]$MockIsaac = $false,
  [switch]$SkipPatchSync = $false,
  [switch]$StartTeleop = $false,
  [int]$TeleopFrequency = 20,
  [switch]$StartSonicServer = $false,
  [string]$SonicPythonExe = "",
  [string]$SonicModelDir = "",
  [switch]$StartKeyboardPlanner = $true,
  [string]$KeyboardPlannerDir = "",
  [string]$KeyboardPlannerInterface = "sim",
  [string]$KeyboardPlannerInputType = "keyboard",
  [string]$KeyboardPlannerOutputType = "ros2",
  [string]$KeyboardPlannerExtraArgs = "",
  [switch]$KeyboardPlannerAutoApprove = $false,
  [switch]$KeyboardPlannerUseWsl = $true,
  [string]$WslDistro = ""
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

function Convert-ToWslPath {
  param([string]$PathValue)
  $resolved = (Resolve-Path $PathValue).Path
  $normalized = $resolved -replace "\\", "/"
  if ($normalized -match "^([A-Za-z]):/(.*)$") {
    $drive = $matches[1].ToLowerInvariant()
    $rest = $matches[2]
    return "/mnt/$drive/$rest"
  }
  throw "Cannot convert Windows path to WSL path: $resolved"
}

function Quote-BashString {
  param([string]$Value)
  if ($Value.Contains("'")) {
    throw "Single quote is not supported in bash argument values: $Value"
  }
  return "'$Value'"
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bridgeScript = Join-Path $root "scripts/start_decoupled_wbc_isaac_bridge.ps1"
if (-not (Test-Path $bridgeScript)) {
  throw "Bridge launcher not found: $bridgeScript"
}

if ([string]::IsNullOrWhiteSpace($DecoupledWbcRoot)) {
  $DecoupledWbcRoot = Join-Path $root "apps/decoupled_wbc_workspace"
} elseif (-not [System.IO.Path]::IsPathRooted($DecoupledWbcRoot)) {
  $DecoupledWbcRoot = Join-Path $root $DecoupledWbcRoot
}
if (-not (Test-Path $DecoupledWbcRoot)) {
  throw "Decoupled WBC root not found: $DecoupledWbcRoot"
}
$DecoupledWbcRoot = (Resolve-Path $DecoupledWbcRoot).Path

if ([string]::IsNullOrWhiteSpace($KeyboardPlannerDir)) {
  $KeyboardPlannerDir = Join-Path $DecoupledWbcRoot "gear_sonic_deploy"
} elseif (-not [System.IO.Path]::IsPathRooted($KeyboardPlannerDir)) {
  $KeyboardPlannerDir = Join-Path $root $KeyboardPlannerDir
}
if ($StartKeyboardPlanner -and -not (Test-Path $KeyboardPlannerDir)) {
  throw "Keyboard planner directory not found: $KeyboardPlannerDir"
}
$deployScript = Join-Path $KeyboardPlannerDir "deploy.sh"
if ($StartKeyboardPlanner -and -not (Test-Path $deployScript)) {
  throw "Keyboard planner deploy script not found: $deployScript"
}

if ($StartTeleop -and $StartKeyboardPlanner) {
  Write-Warning "Both teleop loop and keyboard planner will publish control goals. Prefer enabling only one publisher."
}

# Piping confirmation input makes stdin non-interactive and breaks keyboard planner control.
$useKeyboardPlannerAutoApprove = [bool]$KeyboardPlannerAutoApprove
if ($StartKeyboardPlanner -and $KeyboardPlannerInputType -eq "keyboard" -and $useKeyboardPlannerAutoApprove) {
  Write-Warning "Keyboard planner auto-approve is not compatible with interactive keyboard input. Falling back to manual confirmation."
  $useKeyboardPlannerAutoApprove = $false
}

Write-Host "[start_decoupled_wbc_keyboard_planner] root=$root"
Write-Host "[start_decoupled_wbc_keyboard_planner] bridge_script=$bridgeScript"
Write-Host "[start_decoupled_wbc_keyboard_planner] decoupled_root=$DecoupledWbcRoot"
if ($StartKeyboardPlanner) {
  Write-Host "[start_decoupled_wbc_keyboard_planner] planner_dir=$KeyboardPlannerDir"
  Write-Host "[start_decoupled_wbc_keyboard_planner] planner_interface=$KeyboardPlannerInterface"
  Write-Host "[start_decoupled_wbc_keyboard_planner] planner_input=$KeyboardPlannerInputType"
  Write-Host "[start_decoupled_wbc_keyboard_planner] planner_output=$KeyboardPlannerOutputType"
  Write-Host "[start_decoupled_wbc_keyboard_planner] planner_use_wsl=$KeyboardPlannerUseWsl"
}

$bridgeArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "$bridgeScript",
  "-IsaacSimRoot", "$IsaacSimRoot",
  "-DecoupledPythonExe", "$DecoupledPythonExe",
  "-DecoupledWbcRoot", "$DecoupledWbcRoot",
  "-Namespace", "$Namespace",
  "-RosDomainId", "$RosDomainId",
  "-RmwImplementation", "$RmwImplementation",
  "-InternalStateTopic", "$InternalStateTopic",
  "-InternalCommandTopic", "$InternalCommandTopic",
  "-StatePublishHz", "$StatePublishHz",
  "-ControlFrequency", "$ControlFrequency"
)
if (-not [string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
  $bridgeArgs += @("-IsaacPythonExe", "$IsaacPythonExe")
}
if (-not [string]::IsNullOrWhiteSpace($UsdPath)) {
  $bridgeArgs += @("-UsdPath", "$UsdPath")
}
if ($IsaacGui) { $bridgeArgs += "-IsaacGui" }
if ($MockIsaac) { $bridgeArgs += "-MockIsaac" }
if ($SkipPatchSync) { $bridgeArgs += "-SkipPatchSync" }
if ($StartTeleop) {
  $bridgeArgs += "-StartTeleop"
  $bridgeArgs += @("-TeleopFrequency", "$TeleopFrequency")
}
if ($StartSonicServer) {
  $bridgeArgs += "-StartSonicServer"
  if (-not [string]::IsNullOrWhiteSpace($SonicPythonExe)) {
    $bridgeArgs += @("-SonicPythonExe", "$SonicPythonExe")
  }
  if (-not [string]::IsNullOrWhiteSpace($SonicModelDir)) {
    $bridgeArgs += @("-SonicModelDir", "$SonicModelDir")
  }
}

$procs = @()
try {
  $bridgeProc = Start-Process -FilePath "powershell" -ArgumentList $bridgeArgs -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += [pscustomobject]@{ Name = "decoupled_bridge"; Process = $bridgeProc }
  Write-Host "[start_decoupled_wbc_keyboard_planner] decoupled_bridge pid=$($bridgeProc.Id)"

  if ($StartKeyboardPlanner) {
    Start-Sleep -Seconds 6
    $plannerWorkspace = (Resolve-Path $KeyboardPlannerDir).Path
    $plannerCommand = "./deploy.sh --input-type $KeyboardPlannerInputType --output-type $KeyboardPlannerOutputType $KeyboardPlannerInterface"
    if (-not [string]::IsNullOrWhiteSpace($KeyboardPlannerExtraArgs)) {
      $plannerCommand = "$plannerCommand $KeyboardPlannerExtraArgs"
    }

    if ($KeyboardPlannerUseWsl) {
      $plannerWorkspaceWsl = Convert-ToWslPath -PathValue $plannerWorkspace
      $bashCommand = "set -euo pipefail; cd $(Quote-BashString $plannerWorkspaceWsl); export ROS_DOMAIN_ID=$(Quote-BashString $RosDomainId); export RMW_IMPLEMENTATION=$(Quote-BashString $RmwImplementation); "
      if ($useKeyboardPlannerAutoApprove) {
        $bashCommand += "printf 'Y\n' | $plannerCommand"
      } else {
        $bashCommand += $plannerCommand
      }

      $plannerArgs = @()
      if (-not [string]::IsNullOrWhiteSpace($WslDistro)) {
        $plannerArgs += @("-d", "$WslDistro")
      }
      $plannerArgs += @("--", "bash", "-lc", $bashCommand)
      $plannerProc = Start-Process -FilePath "wsl.exe" -ArgumentList $plannerArgs -WorkingDirectory $root -PassThru
    } else {
      $bash = Get-Command bash -ErrorAction SilentlyContinue
      if ($null -eq $bash) {
        throw "bash command not found. Use -KeyboardPlannerUseWsl or install bash."
      }
      $plannerWorkspacePosix = ($plannerWorkspace -replace "\\", "/")
      $bashCommand = "set -euo pipefail; cd $(Quote-BashString $plannerWorkspacePosix); export ROS_DOMAIN_ID=$(Quote-BashString $RosDomainId); export RMW_IMPLEMENTATION=$(Quote-BashString $RmwImplementation); "
      if ($useKeyboardPlannerAutoApprove) {
        $bashCommand += "printf 'Y\n' | $plannerCommand"
      } else {
        $bashCommand += $plannerCommand
      }
      $plannerProc = Start-Process -FilePath $bash.Source -ArgumentList @("-lc", $bashCommand) -WorkingDirectory $plannerWorkspace -PassThru
    }

    $procs += [pscustomobject]@{ Name = "keyboard_planner"; Process = $plannerProc }
    Write-Host "[start_decoupled_wbc_keyboard_planner] keyboard_planner pid=$($plannerProc.Id)"
  }

  Write-Host "[start_decoupled_wbc_keyboard_planner] running. Press Ctrl+C to stop all child processes."
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
        Write-Warning "[start_decoupled_wbc_keyboard_planner] $($entry.Name) exited (code=$code). stopping remaining processes."
      }
      break
    }
  }
}
finally {
  Write-Host "[start_decoupled_wbc_keyboard_planner] stopping child processes..."
  foreach ($entry in $procs) {
    $proc = $entry.Process
    if ($null -ne $proc) {
      try {
        $proc.Refresh()
        if (-not $proc.HasExited) {
          Stop-Process -Id $proc.Id -Force
          Write-Host "[start_decoupled_wbc_keyboard_planner] stopped $($entry.Name) pid=$($proc.Id)"
        }
      } catch {
        Write-Host "[start_decoupled_wbc_keyboard_planner] stop failed for $($entry.Name): $($_.Exception.Message)"
      }
    }
  }
}
