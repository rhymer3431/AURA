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
  [switch]$SkipControlLoop = $false,
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
  [string]$TensorRtRoot = "",
  [switch]$KeyboardPlannerAutoApprove = $false,
  [switch]$KeyboardPlannerUseWsl = $true,
  [string]$WslDistro = "",
  [switch]$UseIsaacModuleEntrypoint = $true
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

function Convert-ToWslPath {
  param([string]$PathValue)
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    throw "Cannot convert empty path to WSL path."
  }

  $candidate = $PathValue
  if (-not [System.IO.Path]::IsPathRooted($candidate)) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PWD.Path $candidate))
  }

  if (Test-Path $candidate) {
    $resolved = (Resolve-Path $candidate).Path
  } else {
    $resolved = [System.IO.Path]::GetFullPath($candidate)
  }
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

function Normalize-LineEndingsToLf {
  param([string]$PathValue)
  if (-not (Test-Path $PathValue)) {
    return
  }

  $raw = [System.IO.File]::ReadAllText($PathValue)
  if (-not $raw.Contains("`r")) {
    return
  }

  $normalized = $raw -replace "`r`n", "`n"
  $normalized = $normalized -replace "`r", "`n"
  [System.IO.File]::WriteAllText($PathValue, $normalized, [System.Text.UTF8Encoding]::new($false))
  Write-Host "[start_decoupled_wbc_keyboard_planner] normalized line endings to LF: $PathValue"
}

function Ensure-FileFromFallback {
  param(
    [string]$TargetPath,
    [string]$FallbackPath,
    [string]$Label
  )

  if (Test-Path $TargetPath) {
    return $true
  }
  if (-not (Test-Path $FallbackPath)) {
    Write-Warning "[start_decoupled_wbc_keyboard_planner] missing $Label and fallback not found: $FallbackPath"
    return $false
  }

  $targetDir = Split-Path -Parent $TargetPath
  if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
  }

  Copy-Item -Path $FallbackPath -Destination $TargetPath -Force
  Write-Host "[start_decoupled_wbc_keyboard_planner] hydrated $Label from fallback: $FallbackPath -> $TargetPath"
  return $true
}

function Repair-SharedObjectAliasFile {
  param(
    [string]$AliasPath,
    [string]$TargetPath
  )

  if (-not (Test-Path $AliasPath)) {
    return
  }
  if (-not (Test-Path $TargetPath)) {
    return
  }

  $aliasItem = Get-Item $AliasPath -ErrorAction SilentlyContinue
  if ($null -eq $aliasItem) {
    return
  }

  # Some copied Linux symlinks become tiny text files on Windows mounts (e.g. "libddsc.so").
  if ($aliasItem.Length -gt 512) {
    return
  }

  $aliasText = ""
  try {
    $aliasText = [System.IO.File]::ReadAllText($AliasPath).Trim()
  } catch {
    return
  }
  if ([string]::IsNullOrWhiteSpace($aliasText)) {
    return
  }
  if (-not $aliasText.Contains(".so")) {
    return
  }

  Copy-Item -Path $TargetPath -Destination $AliasPath -Force
  Write-Host "[start_decoupled_wbc_keyboard_planner] repaired shared library alias: $AliasPath <= $TargetPath"
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bridgeScript = Join-Path $root "scripts/start_decoupled_wbc_isaac_bridge.ps1"
if (-not (Test-Path $bridgeScript)) {
  throw "Bridge launcher not found: $bridgeScript"
}

$plannerLogDir = Join-Path $root "tmp/planner_logs"
if (-not (Test-Path $plannerLogDir)) {
  New-Item -ItemType Directory -Path $plannerLogDir -Force | Out-Null
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
if ($StartKeyboardPlanner) {
  $lineEndingTargets = @(
    $deployScript,
    (Join-Path $KeyboardPlannerDir "scripts/setup_env.sh"),
    (Join-Path $KeyboardPlannerDir "scripts/install_deps.sh"),
    (Join-Path $KeyboardPlannerDir "scripts/install_ros2_humble.sh"),
    (Join-Path $KeyboardPlannerDir "scripts/restore_ubuntu_sources.sh"),
    (Join-Path $KeyboardPlannerDir "scripts/setup_wifi.sh"),
    (Join-Path $KeyboardPlannerDir "docker/run-ros2-dev.sh")
  )
  foreach ($targetPath in $lineEndingTargets) {
    Normalize-LineEndingsToLf -PathValue $targetPath
  }

  $legacyModelRoot = Join-Path $root "apps/gear_sonic_deploy"
  $decoderTarget = Join-Path $KeyboardPlannerDir "policy/release/model_decoder.onnx"
  $encoderTarget = Join-Path $KeyboardPlannerDir "policy/release/model_encoder.onnx"
  $plannerTarget = Join-Path $KeyboardPlannerDir "planner/target_vel/V2/planner_sonic.onnx"
  $decoderFallback = Join-Path $legacyModelRoot "model_decoder.onnx"
  $encoderFallback = Join-Path $legacyModelRoot "model_encoder.onnx"
  $plannerFallback = Join-Path $legacyModelRoot "planner_sonic.onnx"

  Ensure-FileFromFallback -TargetPath $decoderTarget -FallbackPath $decoderFallback -Label "model_decoder.onnx" | Out-Null
  Ensure-FileFromFallback -TargetPath $encoderTarget -FallbackPath $encoderFallback -Label "model_encoder.onnx" | Out-Null
  Ensure-FileFromFallback -TargetPath $plannerTarget -FallbackPath $plannerFallback -Label "planner_sonic.onnx" | Out-Null

  $sharedLibAliasPairs = @(
    @{ alias = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddsc.so.0"); target = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddsc.so") },
    @{ alias = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddscxx.so.0"); target = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddscxx.so") },
    @{ alias = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so.0"); target = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so") },
    @{ alias = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so.0"); target = (Join-Path $KeyboardPlannerDir "thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so") }
  )
  foreach ($pair in $sharedLibAliasPairs) {
    Repair-SharedObjectAliasFile -AliasPath $pair.alias -TargetPath $pair.target
  }
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
  if (-not [string]::IsNullOrWhiteSpace($TensorRtRoot)) {
    Write-Host "[start_decoupled_wbc_keyboard_planner] TensorRT_ROOT(override)=$TensorRtRoot"
  }
}

$bridgeArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "$bridgeScript",
  "-DecoupledPythonExe=$DecoupledPythonExe",
  "-DecoupledWbcRoot=$DecoupledWbcRoot",
  "-Namespace=$Namespace",
  "-RosDomainId=$RosDomainId",
  "-RmwImplementation=$RmwImplementation",
  "-InternalStateTopic=$InternalStateTopic",
  "-InternalCommandTopic=$InternalCommandTopic",
  "-StatePublishHz=$StatePublishHz",
  "-ControlFrequency=$ControlFrequency"
)
if (-not [string]::IsNullOrWhiteSpace($IsaacSimRoot)) {
  $bridgeArgs += "-IsaacSimRoot=$IsaacSimRoot"
}
if (-not [string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
  $bridgeArgs += "-IsaacPythonExe=$IsaacPythonExe"
}
if (-not [string]::IsNullOrWhiteSpace($UsdPath)) {
  $bridgeArgs += "-UsdPath=$UsdPath"
}
if ($IsaacGui) { $bridgeArgs += "-IsaacGui" }
if ($MockIsaac) { $bridgeArgs += "-MockIsaac" }
if ($SkipPatchSync) { $bridgeArgs += "-SkipPatchSync" }
if ($SkipControlLoop) { $bridgeArgs += "-SkipControlLoop" }
if ($UseIsaacModuleEntrypoint) { $bridgeArgs += "-UseIsaacModuleEntrypoint" }
if ($StartTeleop) {
  $bridgeArgs += "-StartTeleop"
  $bridgeArgs += "-TeleopFrequency=$TeleopFrequency"
}
if ($StartSonicServer) {
  $bridgeArgs += "-StartSonicServer"
  if (-not [string]::IsNullOrWhiteSpace($SonicPythonExe)) {
    $bridgeArgs += "-SonicPythonExe=$SonicPythonExe"
  }
  if (-not [string]::IsNullOrWhiteSpace($SonicModelDir)) {
    $bridgeArgs += "-SonicModelDir=$SonicModelDir"
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
      $plannerLogPath = Join-Path $plannerLogDir ("keyboard_planner_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
      $plannerLogPathWsl = Convert-ToWslPath -PathValue $plannerLogPath

      $tensorRtRootWsl = ""
      if (-not [string]::IsNullOrWhiteSpace($TensorRtRoot)) {
        if ($TensorRtRoot.StartsWith("/")) {
          $tensorRtRootWsl = $TensorRtRoot
        } elseif ([System.IO.Path]::IsPathRooted($TensorRtRoot)) {
          if (-not (Test-Path $TensorRtRoot)) {
            throw "TensorRtRoot not found: $TensorRtRoot"
          }
          $tensorRtRootWsl = Convert-ToWslPath -PathValue $TensorRtRoot
        } else {
          $tensorRtRootWsl = $TensorRtRoot
        }
      }
      Write-Host "[start_decoupled_wbc_keyboard_planner] planner_log=$plannerLogPath"

      $plannerRuntimeCommand = $plannerCommand
      if ($useKeyboardPlannerAutoApprove) {
        $plannerRuntimeCommand = "printf 'Y\n' | $plannerRuntimeCommand"
      }

      $bashCommand = "set -uo pipefail; "
      $bashCommand += "cd $(Quote-BashString $plannerWorkspaceWsl); "
      $bashCommand += "export ROS_DOMAIN_ID=$(Quote-BashString $RosDomainId); "
      $bashCommand += "export RMW_IMPLEMENTATION=$(Quote-BashString $RmwImplementation); "
      $bashCommand += "planner_log=$(Quote-BashString $plannerLogPathWsl); "
      $bashCommand += 'mkdir -p "${planner_log%/*}"; '
      if (-not [string]::IsNullOrWhiteSpace($tensorRtRootWsl)) {
        $bashCommand += "export TensorRT_ROOT=$(Quote-BashString $tensorRtRootWsl); "
      }
      $bashCommand += 'if [ -n "${TensorRT_ROOT:-}" ] && [ -f "${TensorRT_ROOT}/include/x86_64-linux-gnu/NvInfer.h" ]; then export TensorRT_ROOT="${TensorRT_ROOT}"; fi; '
      $bashCommand += 'if [ -n "${TensorRT_ROOT:-}" ] && [ -f "${TensorRT_ROOT}/include/aarch64-linux-gnu/NvInfer.h" ]; then export TensorRT_ROOT="${TensorRT_ROOT}"; fi; '
      $bashCommand += 'if [ -z "${TensorRT_ROOT:-}" ] || { [ ! -f "${TensorRT_ROOT}/include/NvInfer.h" ] && [ ! -f "${TensorRT_ROOT}/include/x86_64-linux-gnu/NvInfer.h" ] && [ ! -f "${TensorRT_ROOT}/include/aarch64-linux-gnu/NvInfer.h" ]; }; then '
      $bashCommand += 'for c in /usr /usr/local /opt/tensorrt /usr/src/tensorrt; do '
      $bashCommand += 'if [ -f "$c/include/NvInfer.h" ] || [ -f "$c/include/x86_64-linux-gnu/NvInfer.h" ] || [ -f "$c/include/aarch64-linux-gnu/NvInfer.h" ]; then export TensorRT_ROOT="$c"; break; fi; '
      $bashCommand += 'done; fi; '
      $bashCommand += 'echo "[keyboard_planner] log: $planner_log"; '
      $bashCommand += 'if [ -z "${TensorRT_ROOT:-}" ] || { [ ! -f "${TensorRT_ROOT}/include/NvInfer.h" ] && [ ! -f "${TensorRT_ROOT}/include/x86_64-linux-gnu/NvInfer.h" ] && [ ! -f "${TensorRT_ROOT}/include/aarch64-linux-gnu/NvInfer.h" ]; }; then '
      $bashCommand += 'echo "[keyboard_planner] TensorRT not found (checked: ${TensorRT_ROOT:-<unset>}/include/NvInfer.h, ${TensorRT_ROOT:-<unset>}/include/x86_64-linux-gnu/NvInfer.h, ${TensorRT_ROOT:-<unset>}/include/aarch64-linux-gnu/NvInfer.h)." | tee -a "$planner_log"; '
      $bashCommand += 'echo "[keyboard_planner] Install TensorRT in WSL or pass -TensorRtRoot <path>." | tee -a "$planner_log"; '
      $bashCommand += 'read -r -p "Press Enter to close this planner window..." _; exit 90; fi; '
      $bashCommand += "{ $plannerRuntimeCommand; } 2>&1 | tee -a " + '"$planner_log"' + "; status=`${PIPESTATUS[0]}; "
      $bashCommand += 'if [ "$status" -ne 0 ]; then echo "[keyboard_planner] exited with code $status" | tee -a "$planner_log"; echo "[keyboard_planner] log: $planner_log" | tee -a "$planner_log"; read -r -p "Press Enter to close this planner window..." _; exit "$status"; fi'

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
      $exitedProcessIds = @{}
      foreach ($entry in $exited) {
        if ($null -ne $entry.Process) {
          try { $exitedProcessIds[[int]$entry.Process.Id] = $true } catch { }
        }
      }

      foreach ($entry in $exited) {
        $code = "unknown"
        try { $code = [string]$entry.Process.ExitCode } catch { }
        Write-Warning "[start_decoupled_wbc_keyboard_planner] $($entry.Name) exited (code=$code)."
      }

      $plannerExited = $false
      foreach ($entry in $exited) {
        if ($entry.Name -eq "keyboard_planner") {
          $plannerExited = $true
          break
        }
      }

      $procs = @(
        $procs | Where-Object {
          $proc = $_.Process
          if ($null -eq $proc) { return $false }
          try {
            return -not $exitedProcessIds.ContainsKey([int]$proc.Id)
          } catch {
            return $false
          }
        }
      )

      if ($plannerExited -or $procs.Count -eq 0) {
        break
      }

      $nonPlannerExited = $false
      foreach ($entry in $exited) {
        if ($entry.Name -ne "keyboard_planner") {
          $nonPlannerExited = $true
          break
        }
      }
      if ($nonPlannerExited) {
        $plannerStillRunning = $false
        foreach ($entry in $procs) {
          if ($entry.Name -eq "keyboard_planner") {
            $plannerStillRunning = $true
            break
          }
        }
        if ($plannerStillRunning) {
          Write-Warning "[start_decoupled_wbc_keyboard_planner] bridge-side process exited; keeping keyboard planner alive for diagnostics/control."
          continue
        }
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
