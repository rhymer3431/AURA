param(
    [int]$StartupTimeoutSec = 180,
    [switch]$PrintConfigJson,
    [switch]$NoExit,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RuntimeArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
$RunSystemPath = Join-Path $ScriptDir "run_system.ps1"
$PowerShellExe = Join-Path $PSHOME "powershell.exe"

function Get-EnvValue {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        [string]$DefaultValue = ""
    )

    foreach ($Name in $Names) {
        $Value = [Environment]::GetEnvironmentVariable($Name, "Process")
        if (-not [string]::IsNullOrWhiteSpace($Value)) {
            return [string]$Value
        }
    }
    return [string]$DefaultValue
}

function Resolve-CondaBat {
    $Candidates = @(
        "$env:USERPROFILE\miniconda3\condabin\conda.bat",
        "$env:USERPROFILE\anaconda3\condabin\conda.bat",
        "C:\Users\mango\anaconda3\condabin\conda.bat",
        "C:\ProgramData\miniconda3\condabin\conda.bat",
        "C:\ProgramData\anaconda3\condabin\conda.bat"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return [System.IO.Path]::GetFullPath($Candidate)
        }
    }
    return ""
}

function Resolve-PortFromUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$DefaultPort
    )

    try {
        $Parsed = [Uri]$Url
        if ($Parsed.Port -gt 0) {
            return [int]$Parsed.Port
        }
    }
    catch {
    }
    return [int]$DefaultPort
}

function Resolve-HostFromUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [string]$DefaultHost = "127.0.0.1"
    )

    try {
        $Parsed = [Uri]$Url
        if (-not [string]::IsNullOrWhiteSpace($Parsed.Host)) {
            return [string]$Parsed.Host
        }
    }
    catch {
    }
    return [string]$DefaultHost
}

function Test-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [int]$TimeoutSec = 120,
        [System.Diagnostics.Process]$Process = $null
    )

    $Deadline = (Get-Date).AddSeconds([Math]::Max($TimeoutSec, 1))
    while ((Get-Date) -lt $Deadline) {
        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                throw "$Name exited before becoming healthy (exit code $($Process.ExitCode))."
            }
        }
        if (Test-HttpReady -Url $Url) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become healthy at $Url within ${TimeoutSec}s."
}

function Start-ComponentProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [ValidateSet("nav", "s2")]
        [string]$Component
    )

    Write-Host "[AURA_SYSTEM] starting $Name via run_system.ps1 -Component $Component"
    return Start-Process `
        -FilePath $PowerShellExe `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunSystemPath, "-Component", $Component) `
        -WorkingDirectory $RepoDir `
        -PassThru
}

function Ensure-ServiceReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [ValidateSet("nav", "s2")]
        [string]$Component,
        [Parameter(Mandatory = $true)]
        [string]$HealthUrl,
        [Parameter(Mandatory = $true)]
        [bool]$Autostart,
        [int]$TimeoutSec = 120
    )

    if (Test-HttpReady -Url $HealthUrl) {
        Write-Host "[AURA_SYSTEM] $Name already healthy: $HealthUrl"
        return
    }
    if (-not $Autostart) {
        throw "$Name is not reachable at $HealthUrl and autostart is disabled."
    }
    $Process = Start-ComponentProcess -Name $Name -Component $Component
    Wait-HttpReady -Url $HealthUrl -Name $Name -TimeoutSec $TimeoutSec -Process $Process
}

function Get-BoolSetting {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [bool]$DefaultValue = $true
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return [bool]$DefaultValue
    }
    $Normalized = ([string]$Value).Trim().ToLowerInvariant()
    return (-not (@("0", "false", "no", "off") -contains $Normalized))
}

if (-not (Test-Path -LiteralPath $RunSystemPath)) {
    throw "run_system.ps1 not found: $RunSystemPath"
}

$CondaEnvName = Get-EnvValue -Names @("CONDA_ENV_NAME", "AURA_CONDA_ENV") -DefaultValue "fa2-cu130-py312"
$CondaBat = Get-EnvValue -Names @("CONDA_BAT") -DefaultValue ""
if ([string]::IsNullOrWhiteSpace($CondaBat)) {
    $CondaBat = Resolve-CondaBat
}
$IsaacSimPath = Get-EnvValue -Names @("ISAACSIM_PATH") -DefaultValue "C:\isaac-sim"
$NavdpUrl = Get-EnvValue -Names @("NAVDP_URL", "G1_POINTGOAL_SERVER_URL") -DefaultValue "http://127.0.0.1:8888"
$InternvlaUrl = Get-EnvValue -Names @("INTERNVLA_URL", "G1_POINTGOAL_SYSTEM2_URL") -DefaultValue "http://127.0.0.1:15801"
$NavInstruction = Get-EnvValue -Names @("NAV_INSTRUCTION") -DefaultValue "Navigate safely to the target and stop when complete."
$NavInstructionLanguage = Get-EnvValue -Names @("NAV_INSTRUCTION_LANGUAGE") -DefaultValue "auto"
$NavCommandApiHost = Get-EnvValue -Names @("NAV_COMMAND_API_HOST") -DefaultValue "127.0.0.1"
$NavCommandApiPort = [int](Get-EnvValue -Names @("NAV_COMMAND_API_PORT") -DefaultValue "8892")
$CameraApiHost = Get-EnvValue -Names @("CAMERA_API_HOST") -DefaultValue "127.0.0.1"
$CameraApiPort = [int](Get-EnvValue -Names @("CAMERA_API_PORT") -DefaultValue "8891")
$CameraPitchDeg = [double](Get-EnvValue -Names @("CAMERA_PITCH_DEG") -DefaultValue "0.0")
$NavdpAutostart = Get-BoolSetting -Value (Get-EnvValue -Names @("NAVDP_AUTOSTART") -DefaultValue "1")
$InternvlaAutostart = Get-BoolSetting -Value (Get-EnvValue -Names @("INTERNVLA_AUTOSTART") -DefaultValue "1")

$NavdpPort = Resolve-PortFromUrl -Url $NavdpUrl -DefaultPort 8888
$InternvlaPort = Resolve-PortFromUrl -Url $InternvlaUrl -DefaultPort 15801
$InternvlaHost = Resolve-HostFromUrl -Url $InternvlaUrl -DefaultHost "127.0.0.1"
$NavdpHealthUrl = "$($NavdpUrl.TrimEnd('/'))/healthz"
$InternvlaHealthUrl = "$($InternvlaUrl.TrimEnd('/'))/healthz"

$Config = [ordered]@{
    conda_env_name = $CondaEnvName
    conda_bat = $CondaBat
    isaacsim_path = $IsaacSimPath
    navdp_url = $NavdpUrl
    internvla_url = $InternvlaUrl
    navdp_port = $NavdpPort
    internvla_host = $InternvlaHost
    internvla_port = $InternvlaPort
    nav_instruction = $NavInstruction
    nav_instruction_language = $NavInstructionLanguage
    nav_command_api_host = $NavCommandApiHost
    nav_command_api_port = $NavCommandApiPort
    camera_api_host = $CameraApiHost
    camera_api_port = $CameraApiPort
    camera_pitch_deg = $CameraPitchDeg
    navdp_autostart = $NavdpAutostart
    internvla_autostart = $InternvlaAutostart
    navdp_health_url = $NavdpHealthUrl
    internvla_health_url = $InternvlaHealthUrl
    runtime_args = @()
}

$ForwardRuntimeArgs = @()
foreach ($Arg in @($RuntimeArgs)) {
    if ($null -eq $Arg) {
        continue
    }
    $StringArg = [string]$Arg
    if ([string]::IsNullOrWhiteSpace($StringArg)) {
        continue
    }
    $ForwardRuntimeArgs += $StringArg
}
$Config["runtime_args"] = @($ForwardRuntimeArgs)

if ($PrintConfigJson) {
    Write-Host ($Config | ConvertTo-Json -Compress -Depth 6)
    if ($NoExit) {
        return 0
    }
    exit 0
}

$env:CONDA_ENV_NAME = $CondaEnvName
if (-not [string]::IsNullOrWhiteSpace($CondaBat)) {
    $env:CONDA_BAT = $CondaBat
}
$env:ISAACSIM_PATH = $IsaacSimPath
$env:NAVDP_URL = $NavdpUrl
$env:INTERNVLA_URL = $InternvlaUrl
$env:NAVDP_PORT = [string]$NavdpPort
$env:INTERNVLA_HOST = $InternvlaHost
$env:INTERNVLA_PORT = [string]$InternvlaPort
$env:NAV_INSTRUCTION = $NavInstruction
$env:NAV_INSTRUCTION_LANGUAGE = $NavInstructionLanguage
$env:NAV_COMMAND_API_HOST = $NavCommandApiHost
$env:NAV_COMMAND_API_PORT = [string]$NavCommandApiPort
$env:CAMERA_API_HOST = $CameraApiHost
$env:CAMERA_API_PORT = [string]$CameraApiPort
$env:CAMERA_PITCH_DEG = [string]$CameraPitchDeg
$env:G1_POINTGOAL_SERVER_URL = $NavdpUrl
$env:G1_POINTGOAL_SYSTEM2_URL = $InternvlaUrl

Ensure-ServiceReady -Name "NavDP server" -Component "nav" -HealthUrl $NavdpHealthUrl -Autostart $NavdpAutostart -TimeoutSec $StartupTimeoutSec
Ensure-ServiceReady -Name "InternVLA server" -Component "s2" -HealthUrl $InternvlaHealthUrl -Autostart $InternvlaAutostart -TimeoutSec $StartupTimeoutSec

Write-Host "[AURA_SYSTEM] launching runtime with source-style Windows contract"
& $RunSystemPath -Component runtime @ForwardRuntimeArgs
$ExitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
if ($NoExit) {
    return $ExitCode
}
exit $ExitCode
