$ErrorActionPreference = "Stop"

function Quote-ForSingleQuotedString {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return $Value -replace "'", "''"
}

function Join-PowerShellArrayLiteralArgs {
    param(
        [string[]]$Values
    )

    if ($null -eq $Values -or $Values.Count -eq 0) {
        return ""
    }

    $Quoted = @()
    foreach ($Value in $Values) {
        $Quoted += "'" + (Quote-ForSingleQuotedString $Value) + "'"
    }
    return [string]::Join(", ", $Quoted)
}

function Test-PythonModules {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )

    $ModuleList = [string]::Join(",", $Modules)
    $TempScript = Join-Path ([System.IO.Path]::GetTempPath()) ("aura_dashboard_module_check_{0}.py" -f [System.Guid]::NewGuid().ToString("N"))
    $PythonCode = @"
import importlib.util
import sys

missing = [name for name in '$ModuleList'.split(',') if name and importlib.util.find_spec(name) is None]
sys.exit(0 if len(missing) == 0 else 1)
"@
    try {
        Set-Content -Path $TempScript -Value $PythonCode -Encoding UTF8
        $Process = Start-Process `
            -FilePath $PythonPath `
            -ArgumentList @($TempScript) `
            -Wait `
            -PassThru `
            -NoNewWindow
        return $Process.ExitCode -eq 0
    }
    finally {
        Remove-Item -Path $TempScript -Force -ErrorAction SilentlyContinue
    }
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$TimeoutSec = 20,
        [System.Diagnostics.Process]$Process = $null
    )

    $Deadline = (Get-Date).AddSeconds([Math]::Max($TimeoutSec, 1))
    do {
        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($null -ne $Response -and $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 300) {
                return $true
            }
        }
        catch {
        }

        if ($null -ne $Process) {
            $Process.Refresh()
            if ($Process.HasExited) {
                return $false
            }
        }

        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $Deadline)

    return $false
}

function Resolve-BackendHealthUrl {
    param(
        [string[]]$Values
    )

    $HostValue = "127.0.0.1"
    $PortValue = "8095"
    for ($Index = 0; $Index -lt $Values.Count; $Index += 1) {
        $Value = [string]$Values[$Index]
        if ($Value -eq "--host" -and $Index + 1 -lt $Values.Count) {
            $HostValue = [string]$Values[$Index + 1]
            $Index += 1
            continue
        }
        if ($Value.StartsWith("--host=")) {
            $HostValue = $Value.Substring("--host=".Length)
            continue
        }
        if ($Value -eq "--port" -and $Index + 1 -lt $Values.Count) {
            $PortValue = [string]$Values[$Index + 1]
            $Index += 1
            continue
        }
        if ($Value.StartsWith("--port=")) {
            $PortValue = $Value.Substring("--port=".Length)
        }
    }

    if ($HostValue -eq "0.0.0.0" -or $HostValue -eq "localhost" -or $HostValue.Contains(":")) {
        $HostValue = "127.0.0.1"
    }

    return "http://${HostValue}:${PortValue}/api/bootstrap"
}

function Resolve-BackendBinding {
    param(
        [string[]]$Values = @()
    )

    $HostValue = "127.0.0.1"
    $PortValue = 8095
    for ($Index = 0; $Index -lt $Values.Count; $Index += 1) {
        $Value = [string]$Values[$Index]
        if ($Value -eq "--host" -and $Index + 1 -lt $Values.Count) {
            $HostValue = [string]$Values[$Index + 1]
            $Index += 1
            continue
        }
        if ($Value.StartsWith("--host=")) {
            $HostValue = $Value.Substring("--host=".Length)
            continue
        }
        if ($Value -eq "--port" -and $Index + 1 -lt $Values.Count) {
            $PortValue = [int]$Values[$Index + 1]
            $Index += 1
            continue
        }
        if ($Value.StartsWith("--port=")) {
            $PortValue = [int]$Value.Substring("--port=".Length)
        }
    }

    $IsLocal = $HostValue -in @("127.0.0.1", "localhost", "0.0.0.0", "::1")
    return @{
        Host    = $HostValue
        Port    = [int]$PortValue
        IsLocal = [bool]$IsLocal
    }
}

function Get-LogTail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [int]$Tail = 80
    )

    if (-not (Test-Path $Path)) {
        return ""
    }
    try {
        return ((Get-Content -Path $Path -Tail $Tail -ErrorAction Stop) -join [Environment]::NewLine)
    }
    catch {
        return ""
    }
}

function Get-ProcessCommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
        if ($null -eq $ProcessInfo -or $null -eq $ProcessInfo.CommandLine) {
            return ""
        }
        return [string]$ProcessInfo.CommandLine
    }
    catch {
        return ""
    }
}

function Stop-StaleDashboardBackend {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoDir,
        [string[]]$BackendArgs = @()
    )

    $Binding = Resolve-BackendBinding -Values $BackendArgs
    if (-not $Binding.IsLocal) {
        return
    }

    $Port = [int]$Binding.Port
    $ExistingPids = @()
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $ExistingPids = @(
            Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique
        )
    }
    if ($null -eq $ExistingPids -or $ExistingPids.Count -eq 0) {
        return
    }

    foreach ($Pid in @($ExistingPids | Where-Object { $_ -gt 0 })) {
        $CommandLine = Get-ProcessCommandLine -ProcessId ([int]$Pid)
        $IsDashboardBackend = $CommandLine -like "*apps.dashboard_backend_app*" -or $CommandLine -like "*dashboard_backend_app*" -or $CommandLine -like "*$RepoDir*isaac-aura*"
        if (-not $IsDashboardBackend) {
            throw "Port $Port is already in use by pid=$Pid. Refusing to stop a non-dashboard process."
        }
        Write-Host "[AURA_DASHBOARD] stopping stale dashboard backend pid=$Pid on port $Port"
        Stop-Process -Id ([int]$Pid) -Force -ErrorAction Stop
    }

    $Deadline = (Get-Date).AddSeconds(10)
    do {
        $StillListening = $false
        if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
            $StillListening = @(
                Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
            ).Count -gt 0
        }
        if (-not $StillListening) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $Deadline)

    throw "A stale dashboard backend listener on port $Port did not exit in time."
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [switch]$NoNewWindow
    )

    $StartProcessParams = @{
        FilePath     = $FilePath
        ArgumentList = $ArgumentList
        Wait         = $true
        PassThru     = $true
    }
    if ($NoNewWindow.IsPresent) {
        $StartProcessParams["NoNewWindow"] = $true
    }
    return Start-Process @StartProcessParams
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir ".."))
$DashboardDir = Join-Path $RepoDir "dashboard"
$SrcDir = Join-Path $RepoDir "src"
$LogDir = Join-Path $RepoDir ".logs\dashboard"
$PowerShellExe = Join-Path $PSHOME "powershell.exe"
$DefaultPythonExe = "C:\Users\mango\anaconda3\python.exe"
$PythonExe = if ($env:AURA_DASHBOARD_PYTHON_EXE) {
    $env:AURA_DASHBOARD_PYTHON_EXE
} elseif (Test-Path $DefaultPythonExe) {
    $DefaultPythonExe
} else {
    "python"
}
$NpmCmd = if ($env:AURA_DASHBOARD_NPM_CMD) { $env:AURA_DASHBOARD_NPM_CMD } else { "npm.cmd" }
$DefaultCargoPath = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
$CargoCmd = if ($env:AURA_DASHBOARD_CARGO_CMD) {
    $env:AURA_DASHBOARD_CARGO_CMD
} elseif (Get-Command "cargo.exe" -ErrorAction SilentlyContinue) {
    "cargo.exe"
} elseif (Test-Path $DefaultCargoPath) {
    $DefaultCargoPath
} else {
    "cargo.exe"
}
$EntryModule = "apps.dashboard_backend_app"

if (-not (Test-Path $DashboardDir)) {
    throw "Dashboard directory not found: $DashboardDir"
}
if (-not (Test-Path (Join-Path $DashboardDir "package.json"))) {
    throw "dashboard/package.json not found."
}
if (-not (Test-Path $PythonExe) -and -not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Get-Command $NpmCmd -ErrorAction SilentlyContinue)) {
    throw "npm executable not found: $NpmCmd"
}
$CargoCmdIsPath = Test-Path $CargoCmd
$CargoCommandInfo = if ($CargoCmdIsPath) {
    $null
} else {
    Get-Command $CargoCmd -ErrorAction SilentlyContinue
}
if (-not $CargoCmdIsPath -and -not $CargoCommandInfo) {
    throw "Rust cargo executable not found: $CargoCmd. Install the Rust toolchain before running the Tauri dashboard."
}
$CargoCommandToInvoke = $CargoCmd
$CargoBinDir = $null
if ($CargoCmdIsPath) {
    $CargoCommandToInvoke = [System.IO.Path]::GetFullPath($CargoCmd)
    $CargoBinDir = Split-Path -Parent $CargoCommandToInvoke
} elseif (-not [string]::IsNullOrWhiteSpace($CargoCommandInfo.Path)) {
    $CargoCommandToInvoke = $CargoCommandInfo.Path
    $CargoBinDir = Split-Path -Parent $CargoCommandToInvoke
}
if (-not (Test-PythonModules -PythonPath $PythonExe -Modules @("aiohttp", "aiortc", "av", "zmq"))) {
    Write-Host "[AURA_DASHBOARD] installing missing backend Python modules: aiohttp aiortc av pyzmq"
    $InstallProcess = Invoke-NativeCommand -FilePath $PythonExe -ArgumentList @("-m", "pip", "install", "aiohttp", "aiortc", "av", "pyzmq") -NoNewWindow
    if ($InstallProcess.ExitCode -ne 0) {
        throw "Failed to install required backend Python modules into $PythonExe"
    }
}

$RepoDirEsc = Quote-ForSingleQuotedString $RepoDir
$DashboardDirEsc = Quote-ForSingleQuotedString $DashboardDir
$SrcDirEsc = Quote-ForSingleQuotedString $SrcDir
$PythonExeEsc = Quote-ForSingleQuotedString $PythonExe
$NpmCmdEsc = Quote-ForSingleQuotedString $NpmCmd
$CargoCmdEsc = Quote-ForSingleQuotedString $CargoCommandToInvoke
$CargoPathBootstrap = ""
if (-not [string]::IsNullOrWhiteSpace($CargoBinDir)) {
    $CargoBinDirEsc = Quote-ForSingleQuotedString $CargoBinDir
    $CargoPathBootstrap = @"
if (Test-Path '$CargoBinDirEsc') {
    `$env:PATH = '$CargoBinDirEsc' + [System.IO.Path]::PathSeparator + `$env:PATH
}
"@
}
$BackendArgListLiteral = Join-PowerShellArrayLiteralArgs $args
$BackendHealthUrl = Resolve-BackendHealthUrl -Values $args
$BackendReadyTimeoutSec = if ($env:AURA_DASHBOARD_BACKEND_READY_TIMEOUT_SEC) {
    [int]$env:AURA_DASHBOARD_BACKEND_READY_TIMEOUT_SEC
} else {
    60
}
$BackendStdoutLog = Join-Path $LogDir "backend.stdout.log"
$BackendStderrLog = Join-Path $LogDir "backend.stderr.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item -Path $BackendStdoutLog, $BackendStderrLog -Force -ErrorAction SilentlyContinue
Stop-StaleDashboardBackend -RepoDir $RepoDir -BackendArgs @($args)

$BackendCommand = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoDirEsc'
`$PreviousPythonPath = `$env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace(`$PreviousPythonPath)) {
    `$env:PYTHONPATH = '$SrcDirEsc'
} else {
    `$env:PYTHONPATH = '$SrcDirEsc' + [System.IO.Path]::PathSeparator + `$PreviousPythonPath
}
try {
    `$PythonProcess = Start-Process -FilePath '$PythonExeEsc' -ArgumentList @('-m', '$EntryModule'$(
if (-not [string]::IsNullOrWhiteSpace($BackendArgListLiteral)) { ", $BackendArgListLiteral" } else { "" }
)) -Wait -PassThru -NoNewWindow
    exit `$PythonProcess.ExitCode
}
finally {
    if ([string]::IsNullOrWhiteSpace(`$PreviousPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        `$env:PYTHONPATH = `$PreviousPythonPath
    }
}
"@

$DesktopCommand = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$DashboardDirEsc'
$CargoPathBootstrap
if (-not (Test-Path 'node_modules')) {
    `$NpmInstall = Start-Process -FilePath '$NpmCmdEsc' -ArgumentList @('install') -Wait -PassThru -NoNewWindow
    if (`$NpmInstall.ExitCode -ne 0) {
        exit `$NpmInstall.ExitCode
    }
}
`$CargoVersion = Start-Process -FilePath '$CargoCmdEsc' -ArgumentList @('-V') -Wait -PassThru -NoNewWindow
if (`$CargoVersion.ExitCode -ne 0) {
    exit `$CargoVersion.ExitCode
}
`$TauriProcess = Start-Process -FilePath '$NpmCmdEsc' -ArgumentList @('run', 'tauri:dev') -Wait -PassThru -NoNewWindow
exit `$TauriProcess.ExitCode
"@

$BackendProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BackendCommand) `
    -WorkingDirectory $RepoDir `
    -RedirectStandardOutput $BackendStdoutLog `
    -RedirectStandardError $BackendStderrLog `
    -PassThru

Write-Host "[AURA_DASHBOARD] backend pid=$($BackendProcess.Id) url=$($BackendHealthUrl -replace '/api/bootstrap$', '')"
Write-Host "[AURA_DASHBOARD] waiting for backend readiness at $BackendHealthUrl"
Write-Host "[AURA_DASHBOARD] backend logs: $BackendStdoutLog / $BackendStderrLog"

if (-not (Wait-HttpReady -Url $BackendHealthUrl -TimeoutSec $BackendReadyTimeoutSec -Process $BackendProcess)) {
    $BackendProcess.Refresh()
    if (-not $BackendProcess.HasExited) {
        try {
            Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
        }
        catch {
        }
    }
    $StdoutTail = Get-LogTail -Path $BackendStdoutLog
    $StderrTail = Get-LogTail -Path $BackendStderrLog
    $TailMessage = ""
    if (-not [string]::IsNullOrWhiteSpace($StdoutTail)) {
        $TailMessage += "`n[backend stdout tail]`n$StdoutTail"
    }
    if (-not [string]::IsNullOrWhiteSpace($StderrTail)) {
        $TailMessage += "`n[backend stderr tail]`n$StderrTail"
    }
    throw "Dashboard backend did not become ready at $BackendHealthUrl within ${BackendReadyTimeoutSec}s. Logs: $BackendStdoutLog / $BackendStderrLog$TailMessage"
}

$DesktopProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $DesktopCommand) `
    -WorkingDirectory $DashboardDir `
    -PassThru

Write-Host "[AURA_DASHBOARD] backend ready"
Write-Host "[AURA_DASHBOARD] tauri pid=$($DesktopProcess.Id) dev-url=auto-select (default http://127.0.0.1:5173)"
Write-Host "[AURA_DASHBOARD] launched backend and Tauri dashboard in separate PowerShell windows."
