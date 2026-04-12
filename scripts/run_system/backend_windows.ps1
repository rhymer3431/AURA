param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 18095,
    [string]$ApiBaseUrl = "",
    [string]$DevOrigin = "http://127.0.0.1:5173",
    [string]$RuntimeUrl = "",
    [string]$InferenceSystemUrl = "",
    [string]$PlannerSystemUrl = "",
    [string]$NavigationSystemUrl = "",
    [string]$ControlRuntimeUrl = "",
    [string]$WebRtcProxyBase = "",
    [string]$WebRtcRgbFps = "",
    [string]$WebRtcDepthFps = "",
    [string]$WebRtcTelemetryHz = "",
    [string]$WebRtcPollIntervalMs = "",
    [string]$WebRtcEnableDepthTrack = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$resolvedApiBaseUrl = if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_DASHBOARD_API_BASE_URL)) {
        "http://127.0.0.1:$Port"
    } else {
        $env:AURA_DASHBOARD_API_BASE_URL
    }
} else {
    $ApiBaseUrl
}
$resolvedRuntimeUrl = if ([string]::IsNullOrWhiteSpace($RuntimeUrl)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AURA_RUNTIME_URL)) {
        $env:AURA_RUNTIME_URL
    } elseif ([string]::IsNullOrWhiteSpace($env:AURA_RUNTIME_SUPERVISOR_URL)) {
        ""
    } else {
        $env:AURA_RUNTIME_SUPERVISOR_URL
    }
} else {
    $RuntimeUrl
}
$resolvedInferenceSystemUrl = if ([string]::IsNullOrWhiteSpace($InferenceSystemUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_INFERENCE_SYSTEM_URL)) {
        "http://127.0.0.1:15880"
    } else {
        $env:AURA_INFERENCE_SYSTEM_URL
    }
} else {
    $InferenceSystemUrl
}
$resolvedPlannerSystemUrl = if ([string]::IsNullOrWhiteSpace($PlannerSystemUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_PLANNER_SYSTEM_URL)) {
        "http://127.0.0.1:17881"
    } else {
        $env:AURA_PLANNER_SYSTEM_URL
    }
} else {
    $PlannerSystemUrl
}
$resolvedNavigationSystemUrl = if ([string]::IsNullOrWhiteSpace($NavigationSystemUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_NAVIGATION_SYSTEM_URL)) {
        "http://127.0.0.1:17882"
    } else {
        $env:AURA_NAVIGATION_SYSTEM_URL
    }
} else {
    $NavigationSystemUrl
}
$resolvedControlRuntimeUrl = if ([string]::IsNullOrWhiteSpace($ControlRuntimeUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_CONTROL_RUNTIME_URL)) {
        "http://127.0.0.1:8892"
    } else {
        $env:AURA_CONTROL_RUNTIME_URL
    }
} else {
    $ControlRuntimeUrl
}
$resolvedWebRtcProxyBase = if ([string]::IsNullOrWhiteSpace($WebRtcProxyBase)) {
    $env:AURA_WEBRTC_PROXY_BASE
} else {
    $WebRtcProxyBase
}
$resolvedWebRtcRgbFps = if ([string]::IsNullOrWhiteSpace($WebRtcRgbFps)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_RGB_FPS)) { "30" } else { $env:AURA_WEBRTC_RGB_FPS }
} else {
    $WebRtcRgbFps
}
$resolvedWebRtcDepthFps = if ([string]::IsNullOrWhiteSpace($WebRtcDepthFps)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_DEPTH_FPS)) { "15" } else { $env:AURA_WEBRTC_DEPTH_FPS }
} else {
    $WebRtcDepthFps
}
$resolvedWebRtcTelemetryHz = if ([string]::IsNullOrWhiteSpace($WebRtcTelemetryHz)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_TELEMETRY_HZ)) { "15" } else { $env:AURA_WEBRTC_TELEMETRY_HZ }
} else {
    $WebRtcTelemetryHz
}
$resolvedWebRtcPollIntervalMs = if ([string]::IsNullOrWhiteSpace($WebRtcPollIntervalMs)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_POLL_INTERVAL_MS)) { "10" } else { $env:AURA_WEBRTC_POLL_INTERVAL_MS }
} else {
    $WebRtcPollIntervalMs
}
$resolvedWebRtcEnableDepthTrack = if ([string]::IsNullOrWhiteSpace($WebRtcEnableDepthTrack)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_ENABLE_DEPTH_TRACK)) { "false" } else { $env:AURA_WEBRTC_ENABLE_DEPTH_TRACK }
} else {
    $WebRtcEnableDepthTrack
}

$env:PYTHONPATH = "$repoRoot\src;$($env:PYTHONPATH)"

$backendArgs = @(
    "-m", "backend.api.serve_backend",
    "--host", $BindHost,
    "--port", "$Port",
    "--api-base-url", $resolvedApiBaseUrl,
    "--dev-origin", $DevOrigin,
    "--inference-system-url", $resolvedInferenceSystemUrl,
    "--planner-system-url", $resolvedPlannerSystemUrl,
    "--navigation-system-url", $resolvedNavigationSystemUrl,
    "--control-runtime-url", $resolvedControlRuntimeUrl,
    "--webrtc-rgb-fps", $resolvedWebRtcRgbFps,
    "--webrtc-depth-fps", $resolvedWebRtcDepthFps,
    "--webrtc-telemetry-hz", $resolvedWebRtcTelemetryHz,
    "--webrtc-poll-interval-ms", $resolvedWebRtcPollIntervalMs
)

if (-not [string]::IsNullOrWhiteSpace($resolvedRuntimeUrl)) {
    $backendArgs += @("--runtime-url", $resolvedRuntimeUrl)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedWebRtcProxyBase)) {
    $backendArgs += @("--webrtc-proxy-base", $resolvedWebRtcProxyBase)
}

$depthTrackEnabled = @("1", "true", "yes", "on") -contains $resolvedWebRtcEnableDepthTrack.Trim().ToLowerInvariant()
if ($depthTrackEnabled) {
    $backendArgs += "--webrtc-enable-depth-track"
} else {
    $backendArgs += "--webrtc-disable-depth-track"
}

& $Python @backendArgs
