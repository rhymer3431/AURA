param(
  [string]$IsaacSimRoot = "C:\isaac-sim",
  [switch]$SkipControlLoop = $false,
  [switch]$UseIsaacModuleEntrypoint = $true,
  [Parameter(ValueFromRemainingArguments = $true)]
  [object[]]$ForwardArgs = @()
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch { }
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding

$target = Join-Path $PSScriptRoot "start_decoupled_wbc_keyboard_planner.ps1"
if (-not (Test-Path $target)) {
  throw "Launcher not found: $target"
}

$resolvedIsaacSimRoot = $IsaacSimRoot
$resolvedUseIsaacModuleEntrypoint = [bool]$UseIsaacModuleEntrypoint

$normalizedForwardArgs = @()
$skipNext = $false
for ($i = 0; $i -lt $ForwardArgs.Count; $i++) {
  $arg = [string]$ForwardArgs[$i]
  if ($skipNext) {
    $skipNext = $false
    continue
  }

  if ($arg -eq "-IsaacSimRoot" -or $arg -eq "-IsaacSimRoot:") {
    if ($i + 1 -lt $ForwardArgs.Count) {
      $candidate = [string]$ForwardArgs[$i + 1]
      if (-not [string]::IsNullOrWhiteSpace($candidate)) {
        $resolvedIsaacSimRoot = $candidate
      }
      $skipNext = $true
    }
    continue
  }

  if ($arg -match "^-IsaacSimRoot:(.+)$") {
    $candidate = $Matches[1]
    if (-not [string]::IsNullOrWhiteSpace($candidate)) {
      $resolvedIsaacSimRoot = $candidate
    }
    continue
  }
  if ($arg -eq "-IsaacGui") {
    continue
  }
  if ($arg -eq "-UseIsaacModuleEntrypoint") {
    $resolvedUseIsaacModuleEntrypoint = $true
    continue
  }
  if ($arg -match "^-UseIsaacModuleEntrypoint:(.+)$") {
    $candidate = $Matches[1]
    if (-not [string]::IsNullOrWhiteSpace($candidate)) {
      $resolvedUseIsaacModuleEntrypoint = [System.Convert]::ToBoolean($candidate)
    }
    continue
  }

  $normalizedForwardArgs += $arg
}

$launchArgs = @(
  "-IsaacGui",
  "-IsaacSimRoot", "$resolvedIsaacSimRoot"
)
if ($resolvedUseIsaacModuleEntrypoint) {
  $launchArgs += "-UseIsaacModuleEntrypoint"
} else {
  $launchArgs += "-UseIsaacModuleEntrypoint:`$false"
}

if ($SkipControlLoop) {
  $launchArgs += "-SkipControlLoop"
}
if ($null -ne $ForwardArgs -and $normalizedForwardArgs.Count -gt 0) {
  $launchArgs += $normalizedForwardArgs
}

Write-Host "[start_decoupled_wbc_keyboard_planner_gui] launching with $($launchArgs -join ' ')"
$psExe = Join-Path $PSHOME "powershell.exe"
if (-not (Test-Path $psExe)) {
  $psExe = "powershell.exe"
}
$psProc = Start-Process -FilePath $psExe -ArgumentList (@(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", "$target"
) + $launchArgs) -Wait -PassThru -NoNewWindow
$exitCode = [int]$psProc.ExitCode
if ($exitCode -ne 0) {
  exit $exitCode
}
