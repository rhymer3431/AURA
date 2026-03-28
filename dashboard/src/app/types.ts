export type ExecutionMode = "TALK" | "NAV" | "MEM_NAV" | "EXPLORE" | "IDLE";
export type LaunchMode = "gui" | "headless";
export type OnnxDevice = "auto" | "cuda" | "cpu";

export type LocomotionConfigForm = {
  actionScale: string;
  onnxDevice: OnnxDevice;
  cmdMaxVx: string;
  cmdMaxVy: string;
  cmdMaxWz: string;
};

export type SessionForm = {
  launchMode: LaunchMode;
  scenePreset: string;
  viewerEnabled: boolean;
  memoryStore: boolean;
  detectionEnabled: boolean;
  locomotionConfig: LocomotionConfigForm;
};

export type ProcessRecord = {
  name: string;
  state: string;
  required: boolean;
  pid: number | null;
  exitCode: number | null;
  startedAt: number | null;
  healthUrl: string;
  stdoutLog: string;
  stderrLog: string;
};

export type ServiceSnapshot = {
  name: string;
  status: string;
  healthUrl?: string;
  latencyMs?: number;
  health?: Record<string, unknown>;
  debug?: Record<string, unknown>;
};

export type System2OutputSnapshot = {
  rawText: string;
  reason: string;
  decisionMode: string;
  needsRequery: boolean;
  historyFrameIds: number[];
  requestedStop: boolean;
  effectiveStop: boolean;
  instruction: string;
  latencyMs?: number;
};

export type System2ServiceSnapshot = {
  name: string;
  status: string;
  healthUrl?: string;
  latencyMs?: number;
  output: System2OutputSnapshot | null;
};

export type ArchitectureNode = {
  name: string;
  status: string;
  summary: string;
  detail: string;
  required: boolean;
  latencyMs?: number | null;
  metrics: Record<string, unknown>;
};

export type MainControlServerNode = ArchitectureNode & {
  core: {
    worldStateStore: ArchitectureNode;
    decisionEngine: ArchitectureNode;
    plannerCoordinator: ArchitectureNode;
    commandResolver: ArchitectureNode;
    safetySupervisor: ArchitectureNode;
  };
};

export type DashboardArchitecture = {
  gateway: ArchitectureNode;
  mainControlServer: MainControlServerNode;
  modules: {
    perception: ArchitectureNode;
    memory: ArchitectureNode;
    s2: ArchitectureNode;
    nav: ArchitectureNode;
    locomotion: ArchitectureNode;
    telemetry: ArchitectureNode;
  };
};

export type SelectedTargetSummary = {
  className: string;
  trackId: string;
  bbox?: [number, number, number, number];
  confidence?: number;
  depthM?: number;
  navGoalPixel?: [number, number];
  worldPose?: [number, number, number];
  source: string;
};

export type LatencyBreakdown = {
  frameAgeMs: number | null;
  perceptionLatencyMs: number | null;
  memoryLatencyMs: number | null;
  s2LatencyMs: number | null;
  navLatencyMs: number | null;
  locomotionLatencyMs: number | null;
};

export type CognitionTraceRecord = {
  timestamp?: number | null;
  frameId: number;
  taskId: string;
  mode: string;
  detectionCount: number;
  trackedDetectionCount: number;
  selectedTarget: string;
  memoryObjectCount: number;
  memoryPlaceCount: number;
  s2RawText: string;
  s2DecisionMode: string;
  s2NeedsRequery: boolean;
  system2PixelGoal: [number, number] | null;
  planVersion: number;
  goalVersion: number;
  trajVersion: number;
  activeCommandType: string;
  actionStatus: string;
  actionReason?: string;
  recoveryState: string;
  recoveryReason: string;
};

export type RecoveryTransition = {
  from: string;
  to: string;
  reason: string;
  timestamp?: number | null;
  retryCount: number;
};

export type DashboardState = {
  timestamp?: number;
  session: {
    active: boolean;
    startedAt: number | null;
    config: {
      launchMode: LaunchMode;
      scenePreset: string;
      viewerEnabled: boolean;
      memoryStore: boolean;
      detectionEnabled: boolean;
      locomotionConfig: {
        actionScale: number;
        onnxDevice: OnnxDevice;
        cmdMaxVx: number;
        cmdMaxVy: number;
        cmdMaxWz: number;
      };
    } | null;
    lastEvent: LogRecord | null;
  };
  processes: ProcessRecord[];
  runtime: Record<string, unknown>;
  sensors: Record<string, unknown>;
  perception: Record<string, unknown>;
  memory: Record<string, unknown>;
  architecture: DashboardArchitecture;
  services: {
    navdp?: ServiceSnapshot;
    dual?: ServiceSnapshot;
    system2?: System2ServiceSnapshot;
  };
  transport: Record<string, unknown>;
  logs: LogRecord[];
  selectedTargetSummary: SelectedTargetSummary | null;
  latencyBreakdown: LatencyBreakdown;
  cognitionTrace: CognitionTraceRecord[];
  recoveryTransitions: RecoveryTransition[];
};

export type BootstrapData = {
  executionModes: ExecutionMode[];
  launchModes: LaunchMode[];
  scenePresets: string[];
  apiBaseUrl: string;
  devOrigin: string;
  webrtcBasePath: string;
};

export type LogRecord = {
  source: string;
  stream: string;
  level?: string;
  message: string;
  path?: string;
  timestampNs?: number;
  details?: Record<string, unknown>;
};

export type NumericSeries = Array<{ t: number; v: number }>;

export type DashboardHistory = {
  stale: NumericSeries;
  goalDistance: NumericSeries;
  navLatency: NumericSeries;
  s2Latency: NumericSeries;
};

export type DashboardContextValue = {
  bootstrap: BootstrapData | null;
  state: DashboardState | null;
  history: DashboardHistory;
  form: SessionForm;
  loading: boolean;
  error: string;
  setForm: (updater: Partial<SessionForm>) => void;
  startSession: () => Promise<void>;
  stopSession: () => Promise<void>;
  submitTask: (instruction: string) => Promise<void>;
  cancelTask: () => Promise<void>;
  refresh: () => Promise<void>;
};

export type ViewerStateMessage = Record<string, unknown>;
export type ViewerTelemetryMessage = Record<string, unknown>;
