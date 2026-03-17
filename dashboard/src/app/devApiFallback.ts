type MockApiResponse = {
  status: number;
  headers: Record<string, string>;
  body: string;
  keepAlive?: boolean;
};

const DEFAULT_DEV_ORIGIN = "http://127.0.0.1:5173";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function buildMockLog(nowMs: number, message: string) {
  return {
    source: "vite-dev-server",
    stream: "event",
    level: "warn",
    message,
    timestampNs: Math.round(nowMs * 1_000_000),
  };
}

function buildMockBootstrap(apiBaseUrl: string) {
  const normalizedBase = trimTrailingSlash(apiBaseUrl);
  return {
    plannerModes: ["interactive", "pointgoal"],
    launchModes: ["gui", "headless"],
    scenePresets: ["warehouse", "interioragent", "interior agent kujiale 3"],
    apiBaseUrl: normalizedBase,
    devOrigin: DEFAULT_DEV_ORIGIN,
    webrtcBasePath: `${normalizedBase}/api/webrtc`,
  };
}

function buildMockState(apiBaseUrl: string, nowMs: number) {
  const message = `AURA dashboard backend is not running at ${trimTrailingSlash(apiBaseUrl)}. Vite dev fallback responses are active.`;
  const eventLog = buildMockLog(nowMs, message);

  return {
    timestamp: nowMs / 1000,
    session: {
      active: false,
      startedAt: null,
      config: null,
      lastEvent: eventLog,
    },
    processes: [
      {
        name: "dashboard_backend",
        state: "inactive",
        required: true,
        pid: null,
        exitCode: null,
        startedAt: null,
        healthUrl: `${trimTrailingSlash(apiBaseUrl)}/api/bootstrap`,
        stdoutLog: "",
        stderrLog: "",
      },
    ],
    runtime: {
      plannerControlMode: "idle",
      lastStatusEvent: {
        state: "mock_mode",
        reason: message,
      },
    },
    sensors: {
      rgbAvailable: false,
      depthAvailable: false,
      poseAvailable: false,
      source: "vite-dev-fallback",
      frameId: "mock",
    },
    perception: {
      detectorReady: false,
      detectorBackend: "mock",
      detectorSelectedReason: "dashboard backend unavailable",
      detectionCount: 0,
      trackedDetectionCount: 0,
      trajectoryPointCount: 0,
      detectorCapability: {
        status: "inactive",
        backend_name: "mock",
      },
    },
    memory: {
      memoryAwareTaskActive: false,
      objectCount: 0,
      placeCount: 0,
      semanticRuleCount: 0,
      scratchpad: {
        taskState: "idle",
        instruction: "Start scripts/powershell/run_dashboard.ps1 for the full backend.",
        nextPriority: "launch backend",
      },
    },
    services: {
      navdp: {
        name: "navdp",
        status: "inactive",
        healthUrl: `${trimTrailingSlash(apiBaseUrl)}/health`,
      },
      dual: {
        name: "dual",
        status: "inactive",
        healthUrl: `${trimTrailingSlash(apiBaseUrl)}/health`,
      },
      system2: {
        name: "system2",
        state: "inactive",
      },
    },
    transport: {
      viewerEnabled: false,
      frameAgeMs: null,
      frameSeq: null,
      frameAvailable: false,
      peerActive: false,
      peerSessionId: null,
      peerTrackRoles: [],
      busHealth: {
        control_endpoint: "tcp://127.0.0.1:5580",
        telemetry_endpoint: "tcp://127.0.0.1:5581",
      },
    },
    logs: [eventLog],
  };
}

function jsonResponse(body: unknown, status = 200): MockApiResponse {
  return {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
    body: JSON.stringify(body),
  };
}

export function shouldBypassApiProxy(requestUrl: string): boolean {
  const pathname = new URL(requestUrl, DEFAULT_DEV_ORIGIN).pathname;
  return pathname.startsWith("/api/");
}

export function buildDevApiFallbackResponse({
  apiBaseUrl,
  method,
  requestUrl,
}: {
  apiBaseUrl: string;
  method?: string;
  requestUrl: string;
}): MockApiResponse {
  const httpMethod = String(method ?? "GET").toUpperCase();
  const parsedUrl = new URL(requestUrl, DEFAULT_DEV_ORIGIN);
  const nowMs = Date.now();
  const normalizedBase = trimTrailingSlash(apiBaseUrl);

  if (httpMethod === "GET" && parsedUrl.pathname === "/api/bootstrap") {
    return jsonResponse(buildMockBootstrap(normalizedBase));
  }

  if (httpMethod === "GET" && parsedUrl.pathname === "/api/state") {
    return jsonResponse(buildMockState(normalizedBase, nowMs));
  }

  if (httpMethod === "GET" && parsedUrl.pathname === "/api/logs") {
    return jsonResponse({
      logs: [
        buildMockLog(
          nowMs,
          `AURA dashboard backend is not running at ${normalizedBase}. Returning Vite mock logs instead.`,
        ),
      ],
    });
  }

  if (httpMethod === "GET" && parsedUrl.pathname === "/api/occupancy/current") {
    const scenePreset = parsedUrl.searchParams.get("scenePreset") ?? "warehouse";
    return jsonResponse({
      available: false,
      scenePreset,
      reason: "dashboard backend is not running, so occupancy metadata is unavailable in Vite mock mode.",
    });
  }

  if (httpMethod === "GET" && parsedUrl.pathname === "/api/events") {
    return {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
      body: `event: state\ndata: ${JSON.stringify(buildMockState(normalizedBase, nowMs))}\n\n`,
      keepAlive: true,
    };
  }

  return jsonResponse(
    {
      error: `AURA dashboard backend is not running at ${normalizedBase}. Start scripts/powershell/run_dashboard.ps1 or point VITE_AURA_API_BASE/AURA_DASHBOARD_PROXY_TARGET to a live backend.`,
      mock: true,
    },
    503,
  );
}
