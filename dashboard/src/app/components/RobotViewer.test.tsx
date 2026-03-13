import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { RobotViewer } from "./RobotViewer";

const mockDashboard = {
  bootstrap: { plannerModes: ["interactive"], launchModes: ["gui"], scenePresets: ["warehouse"], devOrigin: "", webrtcBasePath: "/api/webrtc" },
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
        showDepth: true,
        memoryStore: true,
        detectionEnabled: true,
      },
      lastEvent: null,
    },
    processes: [],
    runtime: {},
    sensors: { source: "aura_runtime" },
    perception: { detectorBackend: "stub" },
    memory: {},
    services: {},
    transport: { viewerEnabled: true, frameAgeMs: 14, peerTrackRoles: ["rgb", "depth"], peerSessionId: "peer-1" },
    logs: [],
  },
  history: { stale: [], goalDistance: [], navdpLatency: [], dualLatency: [] },
  form: {
    plannerMode: "interactive" as const,
    launchMode: "gui" as const,
    scenePreset: "warehouse",
    viewerEnabled: true,
    showDepth: true,
    memoryStore: true,
    detectionEnabled: true,
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
  connected: true,
  error: "",
  session: { type: "session_ready", trackRoles: ["rgb", "depth"] },
  snapshot: {
    type: "snapshot",
    source: "aura_runtime",
    image: { width: 320, height: 180 },
    detector_backend: "stub",
    activeTarget: { nav_goal_pixel: [160, 90] },
  },
  telemetry: {
    type: "frame_meta",
    detections: [{ class_name: "apple", confidence: 0.9, bbox_xyxy: [10, 20, 120, 140] }],
    trajectoryPixels: [[10, 10], [20, 20], [30, 30]],
    activeTarget: { nav_goal_pixel: [160, 90] },
  },
  trackRoles: ["rgb", "depth"],
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
      }),
      configurable: true,
    });

    render(<RobotViewer />);

    expect(screen.getByText("WEBRTC")).toBeInTheDocument();
    expect(screen.getByText("Detected:")).toBeInTheDocument();

    await waitFor(() => {
      expect(strokeRect).toHaveBeenCalled();
      expect(fillText).toHaveBeenCalledWith(expect.stringContaining("apple"), expect.any(Number), expect.any(Number));
      expect(arc).toHaveBeenCalled();
      expect(lineTo).toHaveBeenCalled();
    });
  });
});
