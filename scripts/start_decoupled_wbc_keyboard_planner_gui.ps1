param(
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

Write-Host "[start_decoupled_wbc_keyboard_planner_gui] launching with -IsaacGui"
if ($null -eq $ForwardArgs -or $ForwardArgs.Count -eq 0) {
  & $target -IsaacGui
} else {
  & $target -IsaacGui @ForwardArgs
}
