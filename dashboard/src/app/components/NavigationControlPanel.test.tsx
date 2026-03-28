import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, vi } from "vitest";

import { NavigationControlPanel } from "./NavigationControlPanel";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  Line: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  CartesianGrid: () => <div />,
  Tooltip: () => <div />,
}));

const mockDashboard: any = {
  bootstrap: null,
  state: null,
  history: {
    stale: [{ t: 1, v: 0.5 }],
    goalDistance: [{ t: 1, v: 2.4 }],
    navLatency: [{ t: 1, v: 48 }],
    s2Latency: [{ t: 1, v: 46 }],
  },
  form: {
    launchMode: "gui",
    scenePreset: "warehouse",
    viewerEnabled: true,
    memoryStore: true,
    detectionEnabled: true,
    locomotionConfig: {
      actionScale: "0.5",
      onnxDevice: "auto",
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

function buildState() {
  return {
    timestamp: 1,
    session: { active: true, startedAt: 1, config: null, lastEvent: null },
    processes: [],
    runtime: {
      executionMode: "NAV",
      plannerControlMode: "trajectory",
      plannerControlReason: "route_refresh",
      activeInstruction: "dock at the charging station",
      planVersion: 3,
      goalVersion: 5,
      trajVersion: 8,
      staleSec: 0.4,
      goalDistanceM: 1.25,
      plannerYawDeltaRad: 0.12,
      navTrajectoryWorld: [[1.0, 2.0, 0.0], [1.5, 2.4, 0.0], [1.9, 2.8, 0.0]],
      navTrajectoryPointCount: 3,
      commandVector: [0.18, -0.04, 0.22],
      commandSpeedMps: 0.184,
      recoveryState: "NORMAL",
      recoveryReason: "clear",
      recoveryRetryCount: 0,
      recoveryBackoffUntilNs: 0,
      activeCommandType: "NAV_TO_POSE",
      actionStatus: {
        state: "running",
        reason: "tracking trajectory",
      },
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
      gateway: { name: "Robot Gateway", status: "ok", summary: "frames live", detail: "", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "task active",
        detail: "no recovery override",
        required: true,
        metrics: {
          taskState: "active",
        },
        core: {
          worldStateStore: { name: "World State Store", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          decisionEngine: { name: "Decision Engine", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          plannerCoordinator: { name: "Planner Coordinator", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          commandResolver: { name: "Command Resolver", status: "ok", summary: "tracking trajectory", detail: "", required: true, metrics: {} },
          safetySupervisor: { name: "Safety Supervisor", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
        },
      },
      modules: {
        perception: { name: "Perception", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        memory: { name: "Memory", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        s2: { name: "S2", status: "ok", summary: "Decision pixel goal", detail: "", required: true, metrics: {} },
        nav: { name: "Nav", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        locomotion: { name: "Locomotion", status: "ok", summary: "", detail: "", required: true, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "", detail: "", required: true, metrics: {} },
      },
    },
    services: {
      system2: {
        name: "system2",
        status: "ok",
        latencyMs: 46,
        output: {
          rawText: "120, 80",
          reason: "120, 80",
          decisionMode: "pixel_goal",
          needsRequery: false,
          historyFrameIds: [14, 18, 21],
          requestedStop: false,
          effectiveStop: false,
          instruction: "dock at the charging station",
          latencyMs: 46,
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
      s2LatencyMs: 46,
      navLatencyMs: 48,
      locomotionLatencyMs: null,
    },
    cognitionTrace: [
      {
        timestamp: 1,
        frameId: 21,
        taskId: "task-1",
        mode: "NAV",
        detectionCount: 2,
        trackedDetectionCount: 1,
        selectedTarget: "track-1",
        memoryObjectCount: 3,
        memoryPlaceCount: 2,
        s2RawText: "120, 80",
        s2DecisionMode: "pixel_goal",
        s2NeedsRequery: false,
        system2PixelGoal: [120, 80],
        planVersion: 3,
        goalVersion: 5,
        trajVersion: 8,
        activeCommandType: "NAV_TO_POSE",
        actionStatus: "running",
        actionReason: "tracking trajectory",
        recoveryState: "NORMAL",
        recoveryReason: "clear",
      },
    ],
    recoveryTransitions: [],
  };
}

beforeEach(() => {
  mockDashboard.state = buildState();
});

describe("NavigationControlPanel", () => {
  it("renders the new decision rail and loop timeline sections", () => {
    render(<NavigationControlPanel />);

    expect(screen.getByText("Decision Rail")).toBeInTheDocument();
    expect(screen.getByText("Loop Timeline")).toBeInTheDocument();
    expect(screen.getByText("S2 Decision")).toBeInTheDocument();
    expect(screen.getByText("Motion Decision")).toBeInTheDocument();
    expect(screen.getByText("Resolver / Recovery")).toBeInTheDocument();
    expect(screen.getAllByText("120, 80").length).toBeGreaterThan(0);
    expect(screen.getAllByText("pixel_goal").length).toBeGreaterThan(0);
    expect(screen.getByText("46ms")).toBeInTheDocument();
  });
});
