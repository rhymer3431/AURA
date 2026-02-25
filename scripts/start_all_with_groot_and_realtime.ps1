param(
  [string]$ProjectRoot = "C:\Users\mango\project\isaac-aura",
  [string]$AgentPythonExe = "C:\isaac-sim\python.bat",
  [string]$GrootRepoRoot = "C:\Users\mango\project\Isaac-GR00T-tmp",
  [string]$GrootModelPath = "models\gr00t_n1_6_g1_pnp_apple_to_plate",
  [string]$GrootTrtEnginePath = "models\gr00t_n1_6_g1_pnp_apple_to_plate\trt_fp8\dit_model_fp8.trt",
  [string]$GrootEmbodimentTag = "UNITREE_G1",
  [switch]$NoExitRealtimeWindow = $true
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

if (-not (Test-Path $ProjectRoot)) {
  throw "ProjectRoot not found: $ProjectRoot"
}

$resolvedProjectRoot = (Resolve-Path $ProjectRoot).Path

$startAllCmd = @"
Set-Location -LiteralPath '$resolvedProjectRoot'
.\scripts\start_all.ps1 -IsaacGui -NoInteractive -AgentPythonExe '$AgentPythonExe'
"@

$grootServerCmd = @"
Set-Location -LiteralPath '$resolvedProjectRoot'
python scripts\run_groot_policy_server_fp8.py --groot-repo-root '$GrootRepoRoot' --model-path '$GrootModelPath' --trt-engine-path '$GrootTrtEnginePath' --embodiment-tag '$GrootEmbodimentTag' --use-sim-policy-wrapper
"@

$realtimeCmd = @"
Set-Location -LiteralPath '$resolvedProjectRoot'
.\scripts\start_groot_realtime.ps1
"@

Write-Host "[launcher] starting start_all.ps1..."
$p1 = Start-Process -FilePath "powershell" -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command", $startAllCmd
) -WorkingDirectory $resolvedProjectRoot -PassThru

Start-Sleep -Seconds 2

Write-Host "[launcher] starting run_groot_policy_server_fp8.py..."
$p2 = Start-Process -FilePath "powershell" -ArgumentList @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-Command", $grootServerCmd
) -WorkingDirectory $resolvedProjectRoot -PassThru

Start-Sleep -Seconds 1

$realtimeArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass"
)
if ($NoExitRealtimeWindow) {
  $realtimeArgs += "-NoExit"
}
$realtimeArgs += @(
  "-Command", $realtimeCmd
)

Write-Host "[launcher] opening start_groot_realtime.ps1 in a new terminal window..."
$p3 = Start-Process -FilePath "powershell" -ArgumentList $realtimeArgs -WorkingDirectory $resolvedProjectRoot -PassThru

Write-Host "[launcher] started:"
Write-Host "  start_all.ps1 pid=$($p1.Id)"
Write-Host "  run_groot_policy_server_fp8.py pid=$($p2.Id)"
Write-Host "  start_groot_realtime.ps1 (new terminal) pid=$($p3.Id)"
