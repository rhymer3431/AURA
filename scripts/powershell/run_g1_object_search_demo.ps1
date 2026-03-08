$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$ProcessLogDir = Join-Path $RepoDir "tmp\process_logs"
$BridgeScript = Join-Path $ScriptDir "run_g1_pointgoal.ps1"
$DualServerScript = Join-Path $ScriptDir "run_vlm_dual_server.ps1"

$DualServerUrl = if ($env:G1_OBJECT_SEARCH_DUAL_SERVER_URL) {
    $env:G1_OBJECT_SEARCH_DUAL_SERVER_URL
} else {
    "http://127.0.0.1:8890"
}
$ServerUrl = if ($env:G1_OBJECT_SEARCH_SERVER_URL) { $env:G1_OBJECT_SEARCH_SERVER_URL } else { "http://127.0.0.1:8888" }
$Instruction = if ($env:G1_OBJECT_SEARCH_INSTRUCTION) {
    $env:G1_OBJECT_SEARCH_INSTRUCTION
} else {
    "Find the bright red cube in the warehouse and stop when you reach it."
}
$DemoObjectX = if ($env:G1_OBJECT_SEARCH_DEMO_OBJECT_X) { $env:G1_OBJECT_SEARCH_DEMO_OBJECT_X } else { "5.0" }
$DemoObjectY = if ($env:G1_OBJECT_SEARCH_DEMO_OBJECT_Y) { $env:G1_OBJECT_SEARCH_DEMO_OBJECT_Y } else { "5.0" }
$DemoObjectSize = if ($env:G1_OBJECT_SEARCH_DEMO_OBJECT_SIZE_M) {
    $env:G1_OBJECT_SEARCH_DEMO_OBJECT_SIZE_M
} else {
    "0.25"
}
$ObjectStopRadius = if ($env:G1_OBJECT_SEARCH_OBJECT_STOP_RADIUS_M) {
    $env:G1_OBJECT_SEARCH_OBJECT_STOP_RADIUS_M
} else {
    "0.8"
}
$ForceRuntimeCamera = if ($env:G1_OBJECT_SEARCH_FORCE_RUNTIME_CAMERA) {
    $env:G1_OBJECT_SEARCH_FORCE_RUNTIME_CAMERA
} else {
    "0"
}
$StartupTimeoutSec = if ($env:G1_OBJECT_SEARCH_STARTUP_TIMEOUT_SEC) {
    [int]$env:G1_OBJECT_SEARCH_STARTUP_TIMEOUT_SEC
} else {
    180
}

function Test-LaunchArgPresent {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    foreach ($launchArg in $InputArgs) {
        foreach ($name in $Names) {
            if ($launchArg -eq $name -or $launchArg.StartsWith("$name=")) {
                return $true
            }
        }
    }
    return $false
}

function Get-LaunchArgValue {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$InputArgs,
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$DefaultValue
    )

    $resolved = $DefaultValue
    for ($i = 0; $i -lt $InputArgs.Length; $i++) {
        $launchArg = $InputArgs[$i]
        foreach ($name in $Names) {
            if ($launchArg -eq $name) {
                if (($i + 1) -lt $InputArgs.Length) {
                    $resolved = $InputArgs[$i + 1]
                }
                continue
            }
            if ($launchArg.StartsWith("$name=")) {
                $resolved = $launchArg.Substring($name.Length + 1)
            }
        }
    }
    return $resolved
}

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

if (-not (Test-Path -LiteralPath $BridgeScript)) {
    Write-Host "[G1 Object Search] bridge launcher not found: `"$BridgeScript`""
    exit 1
}

$PlannerModeOverride = Get-LaunchArgValue -InputArgs $args -Names @("--planner-mode") -DefaultValue ""
if ((-not [string]::IsNullOrWhiteSpace($PlannerModeOverride)) -and ($PlannerModeOverride.ToLowerInvariant() -ne "dual")) {
    Write-Host "[G1 Object Search] --planner-mode must remain dual for this demo."
    exit 1
}

$EffectiveDualServerUrl = Get-LaunchArgValue -InputArgs $args -Names @("--dual-server-url") -DefaultValue $DualServerUrl
$EffectiveServerUrl = Get-LaunchArgValue -InputArgs $args -Names @("--server-url") -DefaultValue $ServerUrl
$EffectiveInstruction = Get-LaunchArgValue -InputArgs $args -Names @("--instruction") -DefaultValue $Instruction
$EffectiveDemoObjectX = Get-LaunchArgValue -InputArgs $args -Names @("--demo-object-x") -DefaultValue $DemoObjectX
$EffectiveDemoObjectY = Get-LaunchArgValue -InputArgs $args -Names @("--demo-object-y") -DefaultValue $DemoObjectY
$EffectiveDemoObjectSize = Get-LaunchArgValue -InputArgs $args -Names @("--demo-object-size-m") -DefaultValue $DemoObjectSize
$EffectiveObjectStopRadius = Get-LaunchArgValue -InputArgs $args -Names @("--object-stop-radius-m") -DefaultValue $ObjectStopRadius
$HasForceRuntimeCameraArg = Test-LaunchArgPresent -InputArgs $args -Names @("--force-runtime-camera")

Write-Host "[G1 Object Search] Ensuring dual server is running"
Write-Host "[G1 Object Search] dual-server-url=$EffectiveDualServerUrl"
Ensure-LocalService -Name "VLM Dual Server" -UrlString $EffectiveDualServerUrl -ScriptPath $DualServerScript -TimeoutSec $StartupTimeoutSec

Write-Host "[G1 Object Search] Starting warehouse object-search demo"
Write-Host "[G1 Object Search] server-url=$EffectiveServerUrl"
Write-Host "[G1 Object Search] instruction=`"$EffectiveInstruction`""
Write-Host "[G1 Object Search] demo-object=($EffectiveDemoObjectX, $EffectiveDemoObjectY) size=$EffectiveDemoObjectSize stop-radius=$EffectiveObjectStopRadius"

$BridgeArgs = @(
    "--planner-mode", "dual",
    "--dual-server-url", $EffectiveDualServerUrl,
    "--server-url", $EffectiveServerUrl,
    "--spawn-demo-object",
    "--instruction", $EffectiveInstruction,
    "--demo-object-x", $EffectiveDemoObjectX,
    "--demo-object-y", $EffectiveDemoObjectY,
    "--demo-object-size-m", $EffectiveDemoObjectSize,
    "--object-stop-radius-m", $EffectiveObjectStopRadius
)

if (($ForceRuntimeCamera -notin @("0", "false", "False", "FALSE", "no", "No", "NO")) -and (-not $HasForceRuntimeCameraArg)) {
    $BridgeArgs += @("--force-runtime-camera")
}

& $BridgeScript @BridgeArgs @args
exit $LASTEXITCODE
