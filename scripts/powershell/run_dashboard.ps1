$ErrorActionPreference = "Stop"

function Quote-ForSingleQuotedString {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return $Value -replace "'", "''"
}

function Join-SingleQuotedArgs {
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
    return [string]::Join(" ", $Quoted)
}

function Test-PythonModules {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )

    $ModuleList = [string]::Join(",", $Modules)
    & $PythonPath -c "import importlib.util; import sys; missing=[name for name in '$ModuleList'.split(',') if importlib.util.find_spec(name) is None]; sys.exit(0 if len(missing) == 0 else 1)"
    return $LASTEXITCODE -eq 0
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

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$DashboardDir = Join-Path $RepoDir "dashboard"
$SrcDir = Join-Path $RepoDir "src"
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
# Resolve a concrete cargo path when possible so nested Tauri calls can inherit the toolchain PATH.
$CargoCommandToInvoke = $CargoCmd
$CargoBinDir = $null
if ($CargoCmdIsPath) {
    $CargoCommandToInvoke = [System.IO.Path]::GetFullPath($CargoCmd)
    $CargoBinDir = Split-Path -Parent $CargoCommandToInvoke
} elseif (-not [string]::IsNullOrWhiteSpace($CargoCommandInfo.Path)) {
    $CargoCommandToInvoke = $CargoCommandInfo.Path
    $CargoBinDir = Split-Path -Parent $CargoCommandToInvoke
}
if (-not (Test-PythonModules -PythonPath $PythonExe -Modules @("aiohttp", "aiortc", "av"))) {
    Write-Host "[AURA_DASHBOARD] installing missing backend Python modules: aiohttp aiortc av"
    & $PythonExe -m pip install aiohttp aiortc av
    if ($LASTEXITCODE -ne 0) {
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
$BackendArgText = Join-SingleQuotedArgs $args
$BackendHealthUrl = "http://127.0.0.1:8095/api/bootstrap"

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
    & '$PythonExeEsc' -m $EntryModule $BackendArgText
    exit `$LASTEXITCODE
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
    & '$NpmCmdEsc' install
    if (`$LASTEXITCODE -ne 0) {
        exit `$LASTEXITCODE
    }
}
& '$CargoCmdEsc' -V
if (`$LASTEXITCODE -ne 0) {
    exit `$LASTEXITCODE
}
& '$NpmCmdEsc' run tauri:dev
exit `$LASTEXITCODE
"@

$BackendProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BackendCommand) `
    -WorkingDirectory $RepoDir `
    -PassThru

Write-Host "[AURA_DASHBOARD] backend pid=$($BackendProcess.Id) url=http://127.0.0.1:8095"
Write-Host "[AURA_DASHBOARD] waiting for backend readiness at $BackendHealthUrl"

if (-not (Wait-HttpReady -Url $BackendHealthUrl -TimeoutSec 20 -Process $BackendProcess)) {
    $BackendProcess.Refresh()
    if (-not $BackendProcess.HasExited) {
        try {
            Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
        }
        catch {
        }
    }
    throw "Dashboard backend did not become ready at $BackendHealthUrl. Inspect the backend PowerShell window for startup errors."
}

$DesktopProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $DesktopCommand) `
    -WorkingDirectory $DashboardDir `
    -PassThru

Write-Host "[AURA_DASHBOARD] backend ready"
Write-Host "[AURA_DASHBOARD] tauri pid=$($DesktopProcess.Id) dev-url=auto-select (default http://127.0.0.1:5173)"
Write-Host "[AURA_DASHBOARD] launched backend and Tauri dashboard in separate PowerShell windows."
