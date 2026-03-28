import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { CognitionLoopLane } from "./CognitionLoopLane";

const mockDashboard = {
  bootstrap: null,
  state: {
    timestamp: 0,
    session: { active: true, startedAt: 1, config: null, lastEvent: null },
    processes: [],
    runtime: {
      executionMode: "NAV",
      plannerControlMode: "trajectory",
      plannerControlReason: "route_refresh",
      activeInstruction: "dock",
      planVersion: 4,
      goalVersion: 5,
      trajVersion: 6,
      goalDistanceM: 1.2,
      commandVector: [0.2, 0.0, 0.1],
      activeCommandType: "NAV_TO_POSE",
      actionStatus: { state: "running", reason: "tracking trajectory" },
      recoveryState: "NORMAL",
      recoveryReason: "clear",
      navTrajectoryPointCount: 3,
    },
    sensors: { frameId: 21, source: "unit_test" },
    perception: {
      detectionCount: 2,
      trackedDetectionCount: 1,
      detectorBackend: "stub",
      detectorSelectedReason: "active track",
      trajectoryPointCount: 3,
    },
    memory: {
      objectCount: 3,
      placeCount: 2,
      scratchpad: { taskState: "active", nextPriority: "approach target" },
      memoryAwareTaskActive: true,
    },
    architecture: {
      gateway: { name: "Robot Gateway", status: "ok", summary: "frames live", detail: "rgb/depth/pose", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "task active",
        detail: "interactive",
        required: true,
        metrics: {},
        core: {
          worldStateStore: { name: "World State Store", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          decisionEngine: { name: "Decision Engine", status: "ok", summary: "gating", detail: "", required: true, metrics: {} },
          plannerCoordinator: { name: "Planner Coordinator", status: "ok", summary: "wiring", detail: "", required: true, metrics: {} },
          commandResolver: { name: "Command Resolver", status: "ok", summary: "tracking trajectory", detail: "", required: true, metrics: {} },
          safetySupervisor: { name: "Safety Supervisor", status: "ok", summary: "overrides", detail: "", required: true, metrics: {} },
        },
      },
      modules: {
        perception: { name: "Perception", status: "ok", summary: "detections live", detail: "", required: true, metrics: {} },
        memory: { name: "Memory", status: "ok", summary: "scratchpad ready", detail: "", required: true, metrics: {} },
        s2: { name: "S2", status: "ok", summary: "decision ready", detail: "", required: true, metrics: {} },
        nav: { name: "Nav", status: "ok", summary: "traj v4", detail: "", required: true, metrics: {} },
        locomotion: { name: "Locomotion", status: "ok", summary: "proposal active", detail: "", required: true, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "state mirror active", detail: "", required: true, metrics: {} },
      },
    },
    services: {
      system2: {
        name: "system2",
        status: "ok",
        output: {
          rawText: "120, 80",
          reason: "120, 80",
          decisionMode: "pixel_goal",
          needsRequery: false,
          historyFrameIds: [18, 21],
          requestedStop: false,
          effectiveStop: false,
          instruction: "dock",
        },
      },
    },
    transport: {},
    logs: [],
    selectedTargetSummary: { className: "apple", trackId: "track-1", source: "perception" },
    latencyBreakdown: {
      frameAgeMs: 18,
      perceptionLatencyMs: 12,
      memoryLatencyMs: null,
      s2LatencyMs: 42,
      navLatencyMs: 48,
      locomotionLatencyMs: null,
    },
    cognitionTrace: [],
    recoveryTransitions: [],
  },
  history: { stale: [], goalDistance: [], navLatency: [], s2Latency: [] },
  form: {
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

vi.mock("../state", () => ({
  useDashboard: () => mockDashboard,
}));

describe("CognitionLoopLane", () => {
  it("changes the selected stage when a stage card is clicked", () => {
    const onSelectStageId = vi.fn();

    render(<CognitionLoopLane selectedStageId="gateway" onSelectStageId={onSelectStageId} />);

    fireEvent.click(screen.getByRole("button", { name: /s2/i }));

    expect(onSelectStageId).toHaveBeenCalledWith("s2");
  });
});
