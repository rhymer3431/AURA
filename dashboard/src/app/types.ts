export type PlannerMode = "interactive";
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
  plannerMode: PlannerMode;
  launchMode: LaunchMode;
  scenePreset: string;
  viewerEnabled: boolean;
  memoryStore: boolean;
  detectionEnabled: boolean;
  locomotionConfig: LocomotionConfigForm;
  goalX: string;
  goalY: string;
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

export type DashboardState = {
  timestamp?: number;
  session: {
    active: boolean;
    startedAt: number | null;
    config: {
      plannerMode: PlannerMode;
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
      goal?: { x: number; y: number };
    } | null;
    lastEvent: LogRecord | null;
  };
  processes: ProcessRecord[];
  runtime: Record<string, unknown>;
  sensors: Record<string, unknown>;
  perception: Record<string, unknown>;
  memory: Record<string, unknown>;
  services: {
    navdp?: ServiceSnapshot;
    dual?: ServiceSnapshot;
    system2?: Record<string, unknown>;
  };
  transport: Record<string, unknown>;
  logs: LogRecord[];
};

export type BootstrapData = {
  plannerModes: PlannerMode[];
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
  navdpLatency: NumericSeries;
  dualLatency: NumericSeries;
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
