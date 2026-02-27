param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [string]$IsaacPythonExe = "",
  [string]$UsdPath = "",
  [string]$Namespace = "g1",
  [string]$LogLevel = "INFO",
  [switch]$Gui = $false,
  [switch]$Mock = $false,
  [switch]$PublishImu = $false,
  [switch]$PublishCompressedColor = $false
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

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = "$root;$($env:PYTHONPATH)"
$env:ISAAC_SIM_ROOT = $IsaacSimRoot

if ([string]::IsNullOrWhiteSpace($UsdPath)) {
  $UsdPath = Join-Path $root "g1/g1_d455.usd"
} elseif (-not [System.IO.Path]::IsPathRooted($UsdPath)) {
  $UsdPath = Join-Path $root $UsdPath
}
if (-not (Test-Path $UsdPath)) {
  throw "USD path not found: $UsdPath"
}
$UsdPath = (Resolve-Path $UsdPath).Path

if ([string]::IsNullOrWhiteSpace($IsaacPythonExe)) {
  $candidate = Join-Path $IsaacSimRoot "python.bat"
  if (-not (Test-Path $candidate)) {
    throw "Isaac Sim python launcher not found: $candidate. Pass -IsaacPythonExe explicitly."
  }
  $IsaacPythonExe = $candidate
}
$IsaacPythonExe = Resolve-Executable -Candidate $IsaacPythonExe -Label "IsaacPythonExe"

Write-Host "[start_isaacsim_simple] root=$root"
Write-Host "[start_isaacsim_simple] python=$IsaacPythonExe"
Write-Host "[start_isaacsim_simple] usd=$UsdPath"
Write-Host "[start_isaacsim_simple] namespace=/$($Namespace.Trim('/'))"
Write-Host "[start_isaacsim_simple] gui=$Gui mock=$Mock"

$args = @(
  "-m", "apps.isaacsim_runner",
  "--usd", "$UsdPath",
  "--namespace", "$Namespace",
  "--log-level", "$LogLevel"
)
if ($Gui) { $args += "--gui" }
if ($Mock) { $args += "--mock" }
if ($PublishImu) { $args += "--publish-imu" }
if ($PublishCompressedColor) { $args += "--publish-compressed-color" }

& $IsaacPythonExe @args
