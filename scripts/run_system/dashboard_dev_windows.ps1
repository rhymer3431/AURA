param(
    [string]$BackendUrl = "http://127.0.0.1:18095",
    [string]$BackendScript = "",
    [string]$DashboardRoot = "",
    [string]$Npm = "npm.cmd"
)

$ErrorActionPreference = "Stop"
$dashboardDir = if ([string]::IsNullOrWhiteSpace($DashboardRoot)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_DASHBOARD_ROOT)) {
        "C:\Users\mango\project\AURA\dashboard"
    } else {
        $env:AURA_DASHBOARD_ROOT
    }
} else {
    $DashboardRoot
}
if (!(Test-Path -LiteralPath $dashboardDir)) {
    throw "Dashboard frontend root not found: $dashboardDir"
}

if (-not [string]::IsNullOrWhiteSpace($BackendScript)) {
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $BackendScript) | Out-Null
}

$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$BackendUrl/api/bootstrap" -TimeoutSec 2 | Out-Null
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

Push-Location $dashboardDir
try {
    & $Npm run dev
} finally {
    Pop-Location
}

