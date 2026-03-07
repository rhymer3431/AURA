param(
    [string]$IsaacRoot = "",
    [string]$DiagnosticsPath = ".\tmp\process_logs\live_smoke\diagnostics.json",
    [string]$ArtifactsDir = ".\tmp\process_logs\live_smoke",
    [double]$PreArtifactTimeoutSec = 20.0,
    [double]$PollIntervalSec = 1.0,
    [switch]$ClearCache,
    [switch]$Warmup,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SmokeArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$IsaacRootResolved = if ([string]::IsNullOrWhiteSpace($IsaacRoot)) {
    [System.IO.Path]::GetFullPath($(if ($env:ISAAC_SIM_ROOT) { $env:ISAAC_SIM_ROOT } else { "C:\isaac-sim" }))
} else {
    [System.IO.Path]::GetFullPath($IsaacRoot)
}
$IsaacPython = if ($env:ISAAC_SIM_PYTHON) { $env:ISAAC_SIM_PYTHON } else { Join-Path $IsaacRootResolved "python.bat" }
$ClearCacheScript = Join-Path $IsaacRootResolved "clear_caches.bat"
$WarmupScript = Join-Path $IsaacRootResolved "warmup.bat"
$LogDir = Join-Path $RepoDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $LogDir "live_smoke_${Timestamp}.log"
$StderrLog = Join-Path $LogDir "live_smoke_${Timestamp}.stderr.log"
$ArtifactsDirResolved = if ([System.IO.Path]::IsPathRooted($ArtifactsDir)) {
    [System.IO.Path]::GetFullPath($ArtifactsDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $RepoDir $ArtifactsDir))
}
$DiagnosticsPathResolved = if ([System.IO.Path]::IsPathRooted($DiagnosticsPath)) {
    [System.IO.Path]::GetFullPath($DiagnosticsPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $RepoDir $DiagnosticsPath))
}
$WrapperSummary = Join-Path $ArtifactsDirResolved "wrapper_summary.json"

if (-not (Test-Path -LiteralPath $IsaacRootResolved)) {
    throw "[Live Smoke] Isaac root not found: $IsaacRootResolved"
}
if (-not (Test-Path -LiteralPath $IsaacPython)) {
    throw "[Live Smoke] Isaac python launcher not found: $IsaacPython"
}

if ($ClearCache) {
    if (-not (Test-Path -LiteralPath $ClearCacheScript)) {
        throw "[Live Smoke] clear_caches.bat not found: $ClearCacheScript"
    }
    Write-Host "[Live Smoke] clearing Isaac caches via $ClearCacheScript"
    & $ClearCacheScript
}
if ($Warmup) {
    if (-not (Test-Path -LiteralPath $WarmupScript)) {
        throw "[Live Smoke] warmup.bat not found: $WarmupScript"
    }
    Write-Host "[Live Smoke] warming up Isaac via $WarmupScript"
    & $WarmupScript
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdoutLog) | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactsDirResolved | Out-Null
Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    $ArgumentList = @(
        "-m", "apps.live_smoke_app",
        "--mode", "smoke",
        "--diagnostics-path", $DiagnosticsPathResolved,
        "--artifacts-dir", $ArtifactsDirResolved
    ) + $SmokeArgs

    Write-Host "[Live Smoke] isaac_root=$IsaacRootResolved"
    Write-Host "[Live Smoke] isaac_python=$IsaacPython"
    Write-Host "[Live Smoke] diagnostics=$DiagnosticsPathResolved"
    $Process = Start-Process -FilePath $IsaacPython -ArgumentList $ArgumentList -WorkingDirectory $RepoDir -NoNewWindow -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    $StartedAt = Get-Date
    $TimedOut = $false
    $TimeoutPhase = ""
    $TimeoutBudgetSec = 0.0

    while (-not $Process.HasExited) {
        if (Test-Path -LiteralPath $DiagnosticsPathResolved) {
            $Diagnostics = Get-Content -LiteralPath $DiagnosticsPathResolved -Raw | ConvertFrom-Json
            $CurrentPhase = [string]$Diagnostics.current_phase
            if (-not [string]::IsNullOrWhiteSpace($CurrentPhase)) {
                $Phase = $Diagnostics.phases | Where-Object { $_.name -eq $CurrentPhase } | Select-Object -First 1
                if ($null -ne $Phase -and [string]$Phase.status -eq "running") {
                    $TimeoutSec = [double]$Phase.timeout_sec
                    if ($TimeoutSec -gt 0.0 -and $null -ne $Phase.started_at_s) {
                        $ElapsedSec = ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0) - [double]$Phase.started_at_s
                        if ($ElapsedSec -gt $TimeoutSec) {
                            $TimedOut = $true
                            $TimeoutPhase = $CurrentPhase
                            $TimeoutBudgetSec = $TimeoutSec
                            Stop-Process -Id $Process.Id -Force
                            break
                        }
                    }
                }
            }
        } else {
            $ElapsedStartup = ((Get-Date) - $StartedAt).TotalSeconds
            if ($ElapsedStartup -gt $PreArtifactTimeoutSec) {
                $TimedOut = $true
                $TimeoutPhase = "process_start"
                $TimeoutBudgetSec = $PreArtifactTimeoutSec
                Stop-Process -Id $Process.Id -Force
                break
            }
        }
        Start-Sleep -Milliseconds ([int]([Math]::Max($PollIntervalSec * 1000.0, 100.0)))
    }

    if ($TimedOut) {
        $WrapperPayload = @{
            diagnostics_path = $DiagnosticsPathResolved
            timeout_phase = $TimeoutPhase
            timeout_sec = $TimeoutBudgetSec
            stdout_log = $StdoutLog
            stderr_log = $StderrLog
        }
        $WrapperPayload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $WrapperSummary -Encoding UTF8
        Write-Host "[Live Smoke] timeout phase=$TimeoutPhase timeout_sec=$TimeoutBudgetSec"
        Write-Host "[Live Smoke] stdout_log=$StdoutLog"
        Write-Host "[Live Smoke] stderr_log=$StderrLog"
        exit 124
    }

    $ExitCode = $Process.ExitCode
    Write-Host "[Live Smoke] exit_code=$ExitCode"
    Write-Host "[Live Smoke] stdout_log=$StdoutLog"
    Write-Host "[Live Smoke] stderr_log=$StderrLog"
    Write-Host "[Live Smoke] diagnostics=$DiagnosticsPathResolved"
    exit $ExitCode
}
finally {
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Pop-Location
}
