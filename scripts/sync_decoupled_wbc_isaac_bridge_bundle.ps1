param(
  [string]$DecoupledWbcRoot = ""
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundleRoot = Join-Path $root "apps/isaac_ros2_bridge_bundle"
if (-not (Test-Path $bundleRoot)) {
  throw "Bridge bundle not found: $bundleRoot"
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

Write-Host "[sync_decoupled_wbc_isaac_bridge_bundle] bundle_root=$bundleRoot"
Write-Host "[sync_decoupled_wbc_isaac_bridge_bundle] decoupled_root=$DecoupledWbcRoot"

foreach ($entry in $fileMap) {
  $src = Join-Path $bundleRoot $entry.source
  if (-not (Test-Path $src)) {
    throw "Source file not found: $src"
  }

  $dst = Join-Path $DecoupledWbcRoot $entry.target
  $dstDir = Split-Path -Parent $dst
  if (-not (Test-Path $dstDir)) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
  }
  Copy-Item -Path $src -Destination $dst -Force
  Write-Host "[sync_decoupled_wbc_isaac_bridge_bundle] synced $($entry.source)"
}

Write-Host "[sync_decoupled_wbc_isaac_bridge_bundle] done"
