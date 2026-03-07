$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$EntryModule = "apps.dual_server_app"
$SrcDir = Join-Path $RepoDir "src"
$PathSep = [System.IO.Path]::PathSeparator
$DefaultCondaExe = "C:\Users\mango\anaconda3\Scripts\conda.exe"
$CondaEnv = "fa2-cu130-py312"
$ProcessLogDir = Join-Path $RepoDir "tmp\process_logs"

$ListenHost = if ($env:DUAL_SERVER_HOST) { $env:DUAL_SERVER_HOST } else { "127.0.0.1" }
$Port = if ($env:DUAL_SERVER_PORT) { $env:DUAL_SERVER_PORT } else { "8890" }
$NavDPUrl = if ($env:DUAL_NAVDP_URL) { $env:DUAL_NAVDP_URL } else { "http://127.0.0.1:8888" }
$VLMUrl = if ($env:DUAL_VLM_URL) { $env:DUAL_VLM_URL } else { "http://127.0.0.1:8080" }
$VLMModel = if ($env:DUAL_VLM_MODEL) { $env:DUAL_VLM_MODEL } else { "InternVLA-N1-System2.Q4_K_M.gguf" }
$VLMTemperature = if ($env:DUAL_VLM_TEMPERATURE) { $env:DUAL_VLM_TEMPERATURE } else { "0.2" }
$VLMTopK = if ($env:DUAL_VLM_TOP_K) { $env:DUAL_VLM_TOP_K } else { "40" }
$VLMTopP = if ($env:DUAL_VLM_TOP_P) { $env:DUAL_VLM_TOP_P } else { "0.95" }
$VLMMinP = if ($env:DUAL_VLM_MIN_P) { $env:DUAL_VLM_MIN_P } else { "0.05" }
$VLMRepeatPenalty = if ($env:DUAL_VLM_REPEAT_PENALTY) { $env:DUAL_VLM_REPEAT_PENALTY } else { "1.1" }
$S2Mode = if ($env:DUAL_S2_MODE) { $env:DUAL_S2_MODE } else { "auto" }
$VLMTimeoutSec = if ($env:DUAL_VLM_TIMEOUT_SEC) { $env:DUAL_VLM_TIMEOUT_SEC } else { "35" }
$S2BackoffMaxSec = if ($env:DUAL_S2_BACKOFF_MAX_SEC) { $env:DUAL_S2_BACKOFF_MAX_SEC } else { "30" }
$CondaExe = if ($env:DUAL_CONDA_EXE) { $env:DUAL_CONDA_EXE } else { $DefaultCondaExe }
$StartupTimeoutSec = if ($env:DUAL_STARTUP_TIMEOUT_SEC) { [int]$env:DUAL_STARTUP_TIMEOUT_SEC } else { 180 }

function Test-LocalUrl([Uri]$Uri) {
    if ($null -eq $Uri) {
        return $false
    }

    return $Uri.Host -in @("127.0.0.1", "localhost", "::1")
}

function Wait-TcpReady([Uri]$Uri, [string]$Name, [int]$TimeoutSec, $Process) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $observedExitCode = $null
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $Process) {
            try {
                if ($Process.HasExited) {
                    $observedExitCode = $Process.ExitCode
                }
            } catch {
                if ($null -eq $observedExitCode) {
                    $observedExitCode = "unknown"
                }
            }
        }

        try {
            $client = [System.Net.Sockets.TcpClient]::new()
            $async = $client.BeginConnect($Uri.Host, $Uri.Port, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne(1000)) {
                $client.EndConnect($async)
                $client.Close()
                Write-Host "[$Name] ready at $($Uri.Scheme)://$($Uri.Host):$($Uri.Port)"
                return
            }
            $client.Close()
        } catch {
        }

        Start-Sleep -Milliseconds 500
    }

    if ($null -ne $observedExitCode) {
        throw "$Name did not become ready within ${TimeoutSec}s ($Uri). launcher exit code: $observedExitCode"
    }

    throw "$Name did not become ready within ${TimeoutSec}s ($Uri)."
}

function Start-BackgroundPowerShell([string]$ScriptPath, [string]$Name) {
    New-Item -ItemType Directory -Path $ProcessLogDir -Force | Out-Null

    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $ScriptPath
    )

    $safeName = ($Name -replace '[^A-Za-z0-9_-]', '_')
    $stdoutLog = Join-Path $ProcessLogDir "${safeName}.stdout.log"
    $stderrLog = Join-Path $ProcessLogDir "${safeName}.stderr.log"
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $RepoDir -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    Write-Host "[$Name] started pid=$($proc.Id) script=`"$ScriptPath`""
    Write-Host "[$Name] stdout-log=`"$stdoutLog`" stderr-log=`"$stderrLog`""
    return $proc
}

function Ensure-LocalService([string]$Name, [string]$UrlString, [string]$ScriptPath, [int]$TimeoutSec) {
    $uri = [Uri]$UrlString
    if (-not (Test-LocalUrl $uri)) {
        Write-Host "[$Name] remote endpoint detected; skipping local autostart url=$UrlString"
        return
    }

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "$Name launcher not found: $ScriptPath"
    }

    try {
        Wait-TcpReady -Uri $uri -Name $Name -TimeoutSec 2 -Process $null
        Write-Host "[$Name] already running"
        return
    } catch {
    }

    $proc = Start-BackgroundPowerShell -ScriptPath $ScriptPath -Name $Name
    Wait-TcpReady -Uri $uri -Name $Name -TimeoutSec $TimeoutSec -Process $proc
}

if (-not (Test-Path -LiteralPath $SrcDir)) {
    Write-Host "[VLM Dual Server] src directory not found: `"$SrcDir`""
    exit 1
}

if (-not (Test-Path -LiteralPath $CondaExe)) {
    Write-Host "[VLM Dual Server] conda executable not found: `"$CondaExe`""
    Write-Host "[VLM Dual Server] Set DUAL_CONDA_EXE to your conda.exe path."
    exit 1
}

$NavDPScript = Join-Path $ScriptDir "run_navdp_server.ps1"
$System2Script = Join-Path $ScriptDir "run_internvla_system2.ps1"

Write-Host "[VLM Dual Server] Ensuring upstream servers are running"
Write-Host "[VLM Dual Server] navdp-url=$NavDPUrl"
Write-Host "[VLM Dual Server] vlm-url=$VLMUrl"
Ensure-LocalService -Name "NavDP Server" -UrlString $NavDPUrl -ScriptPath $NavDPScript -TimeoutSec $StartupTimeoutSec
Ensure-LocalService -Name "InternVLA System2" -UrlString $VLMUrl -ScriptPath $System2Script -TimeoutSec $StartupTimeoutSec

Write-Host "[VLM Dual Server] Starting dual orchestrator"
Write-Host "[VLM Dual Server] conda-exe=`"$CondaExe`""
Write-Host "[VLM Dual Server] env=`"$CondaEnv`""
Write-Host "[VLM Dual Server] module=`"$EntryModule`""
Write-Host "[VLM Dual Server] host=$ListenHost port=$Port"
Write-Host "[VLM Dual Server] navdp-url=$NavDPUrl vlm-url=$VLMUrl vlm-model=$VLMModel s2-mode=$S2Mode"
Write-Host "[VLM Dual Server] vlm-timeout-sec=$VLMTimeoutSec s2-backoff-max-sec=$S2BackoffMaxSec"
Write-Host "[VLM Dual Server] vlm-sampling temp=$VLMTemperature top-k=$VLMTopK top-p=$VLMTopP min-p=$VLMMinP repeat-penalty=$VLMRepeatPenalty"

Push-Location $RepoDir
$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcDir
} else {
    $env:PYTHONPATH = "$SrcDir$PathSep$PreviousPythonPath"
}
try {
    & $CondaExe run --no-capture-output -n $CondaEnv python -m $EntryModule --host $ListenHost --port $Port --navdp-url $NavDPUrl --vlm-url $VLMUrl --vlm-model $VLMModel --vlm-temperature $VLMTemperature --vlm-top-k $VLMTopK --vlm-top-p $VLMTopP --vlm-min-p $VLMMinP --vlm-repeat-penalty $VLMRepeatPenalty --s2-mode $S2Mode --vlm-timeout-sec $VLMTimeoutSec --s2-failure-backoff-max-sec $S2BackoffMaxSec @args
    exit $LASTEXITCODE
}
finally {
    if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    Pop-Location
}
