param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacPythonExe = "",
  [switch]$IsaacGui = $false,
  [switch]$MockIsaac = $false,
  [string]$SonicPythonExe = "",
  [string]$SonicModelDir = "",
  [Alias("Host")]
  [string]$SonicHost = "127.0.0.1",
  [int]$SonicPort = 5556,
  [double]$SonicActionScaleMultiplier = 0.10,
  [string]$GrootPythonExe = "python",
  [string]$GrootServerCommand = "",
  [string]$GrootServerCwd = "",
  [string]$GrootRepoRoot = "C:/Users/mango/project/Isaac-GR00T-tmp",
  [string]$GrootModelPath = "models/gr00t_n1_6_g1_pnp_apple_to_plate",
  [string]$GrootTrtEnginePath = "models/gr00t_n1_6_g1_pnp_apple_to_plate/trt_fp8/dit_model_fp8.trt",
  [string]$GrootEmbodimentTag = "UNITREE_G1",
  [switch]$NoSimPolicyWrapper = $false
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"
$env:ISAAC_SIM_ROOT = $IsaacSimRoot

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
$GrootPythonExe = Resolve-Executable -Candidate $GrootPythonExe -Label "GrootPythonExe"

$configPath = Join-Path $root "apps/agent_runtime/config.yaml"
if ([string]::IsNullOrWhiteSpace($GrootServerCommand) -and (Test-Path $configPath)) {
  try {
    $cfgObj = Get-Content $configPath -Raw | ConvertFrom-Json
    $cfgStartCmd = [string]$cfgObj.manipulation.policy_server.start_command
    if (-not [string]::IsNullOrWhiteSpace($cfgStartCmd)) {
      $GrootServerCommand = $cfgStartCmd
    }
  } catch {
    Write-Host "[start_g1_sonic_groot] config parse warning: $($_.Exception.Message)"
  }
}

$resolvedGrootCwd = $root
if (-not [string]::IsNullOrWhiteSpace($GrootServerCwd)) {
  $candidateCwd = if ([System.IO.Path]::IsPathRooted($GrootServerCwd)) {
    $GrootServerCwd
  } else {
    Join-Path $root $GrootServerCwd
  }
  if (-not (Test-Path $candidateCwd)) {
    throw "Invalid -GrootServerCwd path: $candidateCwd"
  }
  $resolvedGrootCwd = (Resolve-Path $candidateCwd).Path
}

$procs = @()
try {
  $g1Args = @(
    "$root/apps/isaacsim_runner/run_headless.py",
    "--usd", "$root/g1/g1_d455.usd",
    "--namespace", "g1",
    "--log-level", "INFO"
  )
  if ($IsaacGui) { $g1Args += "--gui" }
  if ($MockIsaac) { $g1Args += "--mock" }
  $g1Proc = Start-Process -FilePath $IsaacPythonExe -ArgumentList $g1Args -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += [pscustomobject]@{ Name = "g1"; Process = $g1Proc }
  Write-Host "[start_g1_sonic_groot] g1 pid=$($g1Proc.Id) exe=$IsaacPythonExe"

  $sonicLauncher = Join-Path $root "scripts/start_sonic_server.ps1"
  if (-not (Test-Path $sonicLauncher)) {
    throw "SONIC launcher not found: $sonicLauncher"
  }
  $sonicPsArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "$sonicLauncher",
    "-Host", "$SonicHost",
    "-Port", "$SonicPort",
    "-ActionScaleMultiplier", "$SonicActionScaleMultiplier"
  )
  if (-not [string]::IsNullOrWhiteSpace($SonicPythonExe)) {
    $sonicPsArgs += @("-PythonExe", "$SonicPythonExe")
  }
  if (-not [string]::IsNullOrWhiteSpace($SonicModelDir)) {
    $sonicPsArgs += @("-ModelDir", "$SonicModelDir")
  }
  $sonicProc = Start-Process -FilePath "powershell" -ArgumentList $sonicPsArgs -WorkingDirectory $root -PassThru -NoNewWindow
  $procs += [pscustomobject]@{ Name = "gear-sonic"; Process = $sonicProc }
  Write-Host "[start_g1_sonic_groot] gear-sonic pid=$($sonicProc.Id) host=$SonicHost port=$SonicPort"

  if ([string]::IsNullOrWhiteSpace($GrootServerCommand)) {
    $grootArgs = @(
      "$root/scripts/run_groot_policy_server_fp8.py",
      "--groot-repo-root", "$GrootRepoRoot",
      "--model-path", "$GrootModelPath",
      "--trt-engine-path", "$GrootTrtEnginePath",
      "--embodiment-tag", "$GrootEmbodimentTag"
    )
    if (-not $NoSimPolicyWrapper) {
      $grootArgs += "--use-sim-policy-wrapper"
    }
    $grootProc = Start-Process -FilePath $GrootPythonExe -ArgumentList $grootArgs -WorkingDirectory $resolvedGrootCwd -PassThru -NoNewWindow
  } else {
    $grootProc = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $GrootServerCommand) -WorkingDirectory $resolvedGrootCwd -PassThru -NoNewWindow
  }
  $procs += [pscustomobject]@{ Name = "gr00t"; Process = $grootProc }
  Write-Host "[start_g1_sonic_groot] gr00t pid=$($grootProc.Id)"

  Write-Host "[start_g1_sonic_groot] 3 processes are running concurrently. Press Ctrl+C to stop all."
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
        Write-Warning "[start_g1_sonic_groot] $($entry.Name) exited (code=$code). Stopping remaining processes."
      }
      break
    }
  }
}
finally {
  Write-Host "[start_g1_sonic_groot] stopping child processes..."
  foreach ($entry in $procs) {
    $proc = $entry.Process
    if ($null -ne $proc) {
      try {
        $proc.Refresh()
        if (-not $proc.HasExited) {
          Stop-Process -Id $proc.Id -Force
          Write-Host "[start_g1_sonic_groot] stopped $($entry.Name) pid=$($proc.Id)"
        }
      } catch {
        Write-Host "[start_g1_sonic_groot] stop failed for $($entry.Name): $($_.Exception.Message)"
      }
    }
  }
}
