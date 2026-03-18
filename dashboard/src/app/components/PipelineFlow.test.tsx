import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { PipelineFlow } from "./PipelineFlow";

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
          commandResolver: { name: "Command Resolver", status: "ok", summary: "arbitration", detail: "", required: true, metrics: {} },
          safetySupervisor: { name: "Safety Supervisor", status: "ok", summary: "overrides", detail: "", required: true, metrics: {} },
        },
      },
      modules: {
        perception: { name: "Perception", status: "ok", summary: "detections live", detail: "", required: true, metrics: {} },
        memory: { name: "Memory", status: "ok", summary: "scratchpad ready", detail: "", required: true, metrics: {} },
        s2: { name: "S2", status: "ok", summary: "interactive planning active", detail: "", required: true, metrics: {} },
        nav: { name: "Nav", status: "ok", summary: "traj v4", detail: "", required: true, metrics: {} },
        locomotion: { name: "Locomotion", status: "ok", summary: "proposal active", detail: "", required: true, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "state mirror active", detail: "", required: true, metrics: {} },
      },
    },
    services: {},
    transport: {},
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

vi.mock("../state", () => ({
  useDashboard: () => mockDashboard,
}));

describe("PipelineFlow", () => {
  it("renders the new runtime topology without legacy dual/nav labels", () => {
    render(<PipelineFlow />);

    expect(screen.getByText("Runtime Architecture Flow")).toBeInTheDocument();
    expect(screen.getAllByText(/Robot Gateway/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Main Control/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Main Control Server Core/i)).toBeInTheDocument();
    expect(screen.queryByText("Dual Server")).not.toBeInTheDocument();
    expect(screen.queryByText("Dual / S2 Path")).not.toBeInTheDocument();
    expect(screen.queryByText("NavDP Service")).not.toBeInTheDocument();
  });
});
