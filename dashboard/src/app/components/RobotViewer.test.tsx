import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { RobotViewer } from "./RobotViewer";

const mockDashboard = {
  bootstrap: {
    plannerModes: ["interactive"],
    launchModes: ["gui"],
    scenePresets: ["warehouse"],
    apiBaseUrl: "http://127.0.0.1:18095",
    devOrigin: "",
    webrtcBasePath: "http://127.0.0.1:18095/api/webrtc",
  },
  state: {
    timestamp: 0,
    session: {
      active: true,
      startedAt: 1,
      config: {
        plannerMode: "interactive",
        launchMode: "gui",
        scenePreset: "warehouse",
        viewerEnabled: true,
        memoryStore: true,
        detectionEnabled: true,
        locomotionConfig: {
          actionScale: 0.5,
          onnxDevice: "auto" as const,
          cmdMaxVx: 0.5,
          cmdMaxVy: 0.3,
          cmdMaxWz: 0.8,
        },
      },
      lastEvent: null,
    },
    processes: [],
    runtime: {},
    sensors: { source: "aura_runtime" },
    perception: { detectorBackend: "stub" },
    memory: {},
    architecture: {
      gateway: { name: "Robot Gateway", status: "ok", summary: "frames live", detail: "", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "task active",
        detail: "",
        required: true,
        metrics: {},
        core: {
          worldStateStore: { name: "World State Store", status: "ok", summary: "", detail: "", required: true, metrics: {} },
          decisionEngine: { name: "Decision Engine", status: "ok", summary: "", detail: "", required: true, metrics: {} },
          plannerCoordinator: { name: "Planner Coordinator", status: "ok", summary: "", detail: "", required: true, metrics: {} },
          commandResolver: { name: "Command Resolver", status: "ok", summary: "", detail: "", required: true, metrics: {} },
          safetySupervisor: { name: "Safety Supervisor", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        },
      },
      modules: {
        perception: { name: "Perception", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        memory: { name: "Memory", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        s2: { name: "S2", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        nav: { name: "Nav", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        locomotion: { name: "Locomotion", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "", detail: "", required: true, metrics: {} },
      },
    },
    services: {},
    transport: { viewerEnabled: true, frameAgeMs: 14, peerTrackRoles: ["rgb", "depth"], peerSessionId: "peer-1" },
    logs: [],
  },
  history: { stale: [], goalDistance: [], navLatency: [], s2Latency: [] },
  form: {
    plannerMode: "interactive" as const,
    launchMode: "gui" as const,
    scenePreset: "warehouse",
    viewerEnabled: true,
    memoryStore: true,
    detectionEnabled: true,
    locomotionConfig: {
      actionScale: "0.5",
      onnxDevice: "auto" as const,
      cmdMaxVx: "0.5",
      cmdMaxVy: "0.3",
      cmdMaxWz: "0.8",
    },
    goalX: "0",
    goalY: "0",
  },
  loading: false,
  error: "",
  setForm: vi.fn(),
  startSession: vi.fn(),
  stopSession: vi.fn(),
  submitTask: vi.fn(),
  cancelTask: vi.fn(),
  refresh: vi.fn(),
};

const mockHookValue = {
  rgbVideoRef: { current: null },
  depthVideoRef: { current: null },
  snapshotRef: {
    current: {
      type: "snapshot",
      source: "aura_runtime",
      image: { width: 320, height: 180 },
      detector_backend: "stub",
      activeTarget: { nav_goal_pixel: [160, 90] },
      system2PixelGoal: [220, 120],
    },
  },
  telemetryRef: {
    current: {
      type: "frame_meta",
      detections: [{ class_name: "apple", confidence: 0.9, bbox_xyxy: [10, 20, 120, 140] }],
      trajectoryPixels: [[10, 10], [20, 20], [30, 30]],
      activeTarget: { nav_goal_pixel: [160, 90] },
      system2PixelGoal: [220, 120],
    },
  },
  connected: true,
  error: "",
  session: { type: "session_ready", trackRoles: ["rgb", "depth"] },
  trackRoles: ["rgb", "depth"],
  hudVersion: 1,
};

vi.mock("../state", () => ({
  useDashboard: () => mockDashboard,
}));

vi.mock("../hooks/useWebRTCViewer", () => ({
  useWebRTCViewer: () => mockHookValue,
}));

describe("RobotViewer", () => {
  it("renders webrtc state and draws overlay content", async () => {
    const strokeRect = vi.fn();
    const fillText = vi.fn();
    const beginPath = vi.fn();
    const moveTo = vi.fn();
    const lineTo = vi.fn();
    const stroke = vi.fn();
    const arc = vi.fn();
    const clearRect = vi.fn();
    const save = vi.fn();
    const restore = vi.fn();

    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      value: () => ({
        strokeRect,
        fillText,
        beginPath,
        moveTo,
        lineTo,
        stroke,
        arc,
        clearRect,
        save,
        restore,
      }),
      configurable: true,
    });

    render(<RobotViewer />);

    expect(screen.getByText("WEBRTC")).toBeInTheDocument();
    expect(screen.getByText("Detected")).toBeInTheDocument();

    await waitFor(() => {
      expect(strokeRect).toHaveBeenCalled();
      expect(fillText).toHaveBeenCalledWith(expect.stringContaining("apple"), expect.any(Number), expect.any(Number));
      expect(fillText).toHaveBeenCalledWith("S2 GOAL", expect.any(Number), expect.any(Number));
      expect(arc).toHaveBeenCalled();
      expect(lineTo).toHaveBeenCalled();
    });
  });
});
