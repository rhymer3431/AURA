param(
  [string]$PythonExe = "",
  [string]$ModelDir = "",
  [string]$Encoder = "",
  [string]$Decoder = "",
  [string]$Planner = "",
  [Alias("Host")]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 5556,
  [double]$ActionScaleMultiplier = 0.10
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"

function Test-PythonModule {
  param(
    [string]$Exe,
    [string]$ModuleName
  )
  try {
    & $Exe -c "import $ModuleName" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  $candidates = @()
  if (Get-Command python -ErrorAction SilentlyContinue) {
    $candidates += "python"
  }
  $isaacPy = "C:\isaac-sim\python.bat"
  if (Test-Path $isaacPy) {
    $candidates += $isaacPy
  }

  foreach ($cand in $candidates) {
    if (Test-PythonModule -Exe $cand -ModuleName "onnxruntime") {
      $PythonExe = $cand
      break
    }
  }

  if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    throw "No usable python found with onnxruntime. Install onnxruntime in your target python, or pass -PythonExe explicitly."
  }
} elseif (-not (Test-PythonModule -Exe $PythonExe -ModuleName "onnxruntime")) {
  throw "onnxruntime import failed in '$PythonExe'. Try running without -PythonExe or use a python where 'import onnxruntime' succeeds."
}

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
  $ModelDir = Join-Path $root "apps/gear_sonic_deploy"
} elseif (-not [System.IO.Path]::IsPathRooted($ModelDir)) {
  $ModelDir = Join-Path $root $ModelDir
}

if (-not (Test-Path $ModelDir)) {
  throw "SONIC model dir not found: $ModelDir"
}
$ModelDir = (Resolve-Path $ModelDir).Path

if ([string]::IsNullOrWhiteSpace($Encoder)) {
  $Encoder = Join-Path $ModelDir "model_encoder.onnx"
} elseif (-not [System.IO.Path]::IsPathRooted($Encoder)) {
  $Encoder = Join-Path $root $Encoder
}
if ([string]::IsNullOrWhiteSpace($Decoder)) {
  $Decoder = Join-Path $ModelDir "model_decoder.onnx"
} elseif (-not [System.IO.Path]::IsPathRooted($Decoder)) {
  $Decoder = Join-Path $root $Decoder
}
if ([string]::IsNullOrWhiteSpace($Planner)) {
  $Planner = Join-Path $ModelDir "planner_sonic.onnx"
} elseif (-not [System.IO.Path]::IsPathRooted($Planner)) {
  $Planner = Join-Path $root $Planner
}

$required = @($Encoder, $Decoder, $Planner)
foreach ($path in $required) {
  if (-not (Test-Path $path)) {
    throw "Required model file not found: $path"
  }
}

Write-Host "[start_sonic_server] root=$root"
Write-Host "[start_sonic_server] python=$PythonExe"
Write-Host "[start_sonic_server] encoder=$Encoder"
Write-Host "[start_sonic_server] decoder=$Decoder"
Write-Host "[start_sonic_server] planner=$Planner"
Write-Host "[start_sonic_server] bind=${BindHost}:${Port}"
Write-Host "[start_sonic_server] action_scale_multiplier=$ActionScaleMultiplier"

$args = @(
  "$root/sonic_policy_server.py",
  "--encoder", "$Encoder",
  "--decoder", "$Decoder",
  "--planner", "$Planner",
  "--host", "$BindHost",
  "--port", "$Port",
  "--action-scale-multiplier", "$ActionScaleMultiplier"
)

& $PythonExe @args
