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

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir "..\.."))
$DashboardDir = Join-Path $RepoDir "dashboard"
$SrcDir = Join-Path $RepoDir "src"
$PowerShellExe = Join-Path $PSHOME "powershell.exe"
$PythonExe = if ($env:AURA_DASHBOARD_PYTHON_EXE) { $env:AURA_DASHBOARD_PYTHON_EXE } else { "python" }
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
$CargoBinDir = Split-Path -Parent $CargoCmd
$EntryModule = "apps.dashboard_backend_app"

if (-not (Test-Path $DashboardDir)) {
    throw "Dashboard directory not found: $DashboardDir"
}
if (-not (Test-Path (Join-Path $DashboardDir "package.json"))) {
    throw "dashboard/package.json not found."
}
if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Get-Command $NpmCmd -ErrorAction SilentlyContinue)) {
    throw "npm executable not found: $NpmCmd"
}
if (-not (Test-Path $CargoCmd) -and -not (Get-Command $CargoCmd -ErrorAction SilentlyContinue)) {
    throw "Rust cargo executable not found: $CargoCmd. Install the Rust toolchain before running the Tauri dashboard."
}

$RepoDirEsc = Quote-ForSingleQuotedString $RepoDir
$DashboardDirEsc = Quote-ForSingleQuotedString $DashboardDir
$SrcDirEsc = Quote-ForSingleQuotedString $SrcDir
$PythonExeEsc = Quote-ForSingleQuotedString $PythonExe
$NpmCmdEsc = Quote-ForSingleQuotedString $NpmCmd
$CargoCmdEsc = Quote-ForSingleQuotedString $CargoCmd
$CargoBinDirEsc = Quote-ForSingleQuotedString $CargoBinDir
$BackendArgText = Join-SingleQuotedArgs $args

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
if (Test-Path '$CargoBinDirEsc') {
    `$env:PATH = '$CargoBinDirEsc' + [System.IO.Path]::PathSeparator + `$env:PATH
}
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

$DesktopProcess = Start-Process `
    -FilePath $PowerShellExe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $DesktopCommand) `
    -WorkingDirectory $DashboardDir `
    -PassThru

Write-Host "[AURA_DASHBOARD] backend pid=$($BackendProcess.Id) url=http://127.0.0.1:8095"
Write-Host "[AURA_DASHBOARD] tauri pid=$($DesktopProcess.Id) dev-url=http://127.0.0.1:5173"
Write-Host "[AURA_DASHBOARD] launched backend and Tauri dashboard in separate PowerShell windows."
