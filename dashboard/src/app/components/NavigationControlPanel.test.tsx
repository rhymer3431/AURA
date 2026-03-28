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
    navLatency: [],
    s2Latency: [],
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
      activeCommandType: "NAV_TO_POSE",
      actionStatus: {
        state: "running",
        reason: "tracking trajectory",
      },
    },
    sensors: {},
    perception: {},
    memory: {},
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
          commandResolver: { name: "Command Resolver", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
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
  };
}

beforeEach(() => {
  mockDashboard.state = buildState();
});

describe("NavigationControlPanel", () => {
  it("renders structured system2 output details", () => {
    render(<NavigationControlPanel />);

    expect(screen.getByText("System2 Output")).toBeInTheDocument();
    expect(screen.getByText("Trajectory Points")).toBeInTheDocument();
    expect(screen.getByText("(1.00, 2.00, 0.00) -> (1.50, 2.40, 0.00) -> (1.90, 2.80, 0.00)")).toBeInTheDocument();
    expect(screen.getByText("[0.18, -0.04, 0.22]")).toBeInTheDocument();
    expect(screen.getAllByText("120, 80").length).toBeGreaterThan(0);
    expect(screen.getByText("pixel_goal")).toBeInTheDocument();
    expect(screen.getByText("46ms")).toBeInTheDocument();
    expect(screen.getByText("14, 18, 21")).toBeInTheDocument();
    expect(screen.getByText("dock at the charging station")).toBeInTheDocument();
  });

  it("shows awaiting-first-decision fallback when session is active but no output exists", () => {
    mockDashboard.state = {
      ...buildState(),
      services: {
        system2: {
          name: "system2",
          status: "awaiting_first_decision",
          output: null,
        },
      },
    };

    render(<NavigationControlPanel />);

    expect(screen.getAllByText("awaiting first decision").length).toBeGreaterThan(0);
  });

  it("shows session-inactive fallback when the runtime session is down", () => {
    mockDashboard.state = {
      ...buildState(),
      session: { active: false, startedAt: null, config: null, lastEvent: null },
      services: {
        system2: {
          name: "system2",
          status: "inactive",
          output: null,
        },
      },
    };

    render(<NavigationControlPanel />);

    expect(screen.getByText("session inactive")).toBeInTheDocument();
  });
});
