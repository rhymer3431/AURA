import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { LogsReplayWorkspace } from "./LogsReplayWorkspace";

vi.mock("./SystemStatusWidgets", () => ({
  LogsWidget: () => <div>LogsWidget</div>,
}));

const mockDashboard = {
  bootstrap: null,
  state: {
    timestamp: 0,
    session: { active: true, startedAt: 1, config: null, lastEvent: null },
    processes: [],
    runtime: {},
    sensors: {},
    perception: {},
    memory: {},
    architecture: {
      gateway: { name: "Robot Gateway", status: "ok", summary: "", detail: "", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "",
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
    transport: {},
    logs: [],
    selectedTargetSummary: null,
    latencyBreakdown: {
      frameAgeMs: null,
      perceptionLatencyMs: null,
      memoryLatencyMs: null,
      s2LatencyMs: null,
      navLatencyMs: null,
      locomotionLatencyMs: null,
    },
    cognitionTrace: [
      {
        timestamp: 10,
        frameId: 11,
        taskId: "task-a",
        mode: "NAV",
        detectionCount: 2,
        trackedDetectionCount: 1,
        selectedTarget: "apple",
        memoryObjectCount: 2,
        memoryPlaceCount: 1,
        s2RawText: "120, 80",
        s2DecisionMode: "pixel_goal",
        s2NeedsRequery: false,
        system2PixelGoal: [120, 80] as [number, number],
        planVersion: 2,
        goalVersion: 3,
        trajVersion: 4,
        activeCommandType: "NAV_TO_POSE",
        actionStatus: "running",
        actionReason: "tracking trajectory",
        recoveryState: "NORMAL",
        recoveryReason: "clear",
      },
      {
        timestamp: 12,
        frameId: 12,
        taskId: "task-b",
        mode: "NAV",
        detectionCount: 1,
        trackedDetectionCount: 1,
        selectedTarget: "banana",
        memoryObjectCount: 1,
        memoryPlaceCount: 1,
        s2RawText: "stop",
        s2DecisionMode: "stop",
        s2NeedsRequery: false,
        system2PixelGoal: null,
        planVersion: 3,
        goalVersion: 4,
        trajVersion: 5,
        activeCommandType: "STOP",
        actionStatus: "running",
        actionReason: "stop requested",
        recoveryState: "SAFE_STOP",
        recoveryReason: "manual override",
      },
    ],
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

describe("LogsReplayWorkspace", () => {
  it("filters the frame trace table by recovery state and S2 raw output", () => {
    render(<LogsReplayWorkspace />);

    expect(screen.getByText("LogsWidget")).toBeInTheDocument();
    expect(screen.getByText("task-a")).toBeInTheDocument();
    expect(screen.getByText("task-b")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("filter recovery state"), { target: { value: "SAFE_STOP" } });
    fireEvent.change(screen.getByPlaceholderText("search S2 raw output"), { target: { value: "stop" } });

    expect(screen.queryByText("task-a")).not.toBeInTheDocument();
    expect(screen.getByText("task-b")).toBeInTheDocument();
  });
});
