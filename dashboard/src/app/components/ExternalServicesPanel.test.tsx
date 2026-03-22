import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { ExternalServicesPanel } from "./ExternalServicesPanel";

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
      gateway: { name: "Robot Gateway", status: "ok", summary: "frames live", detail: "", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "task active",
        detail: "",
        required: true,
        metrics: {},
        core: {
          worldStateStore: { name: "World State Store", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          decisionEngine: { name: "Decision Engine", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          plannerCoordinator: { name: "Planner Coordinator", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          commandResolver: { name: "Command Resolver", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
          safetySupervisor: { name: "Safety Supervisor", status: "ok", summary: "ready", detail: "", required: true, metrics: {} },
        },
      },
      modules: {
        perception: { name: "Perception", status: "ok", summary: "detections live", detail: "", required: true, metrics: {} },
        memory: { name: "Memory", status: "ok", summary: "scratchpad ready", detail: "", required: true, metrics: {} },
        s2: { name: "S2", status: "ok", summary: "interactive planning active", detail: "", required: true, latencyMs: 130, metrics: {} },
        nav: { name: "Nav", status: "ok", summary: "traj v4", detail: "", required: true, latencyMs: 48, metrics: {} },
        locomotion: { name: "Locomotion", status: "ok", summary: "proposal active", detail: "", required: true, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "state mirror active", detail: "", required: true, metrics: {} },
      },
    },
    services: {},
    transport: {},
    logs: [],
  },
  history: { stale: [], goalDistance: [], navLatency: [{ t: 1, v: 48 }], s2Latency: [{ t: 1, v: 130 }] },
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

describe("ExternalServicesPanel", () => {
  it("renders module-centric health cards and hides legacy service names", () => {
    render(<ExternalServicesPanel />);

    expect(screen.getByText("External Services")).toBeInTheDocument();
    expect(screen.getByText("Main Control Server")).toBeInTheDocument();
    expect(screen.getByText("Robot Gateway")).toBeInTheDocument();
    expect(screen.getByText("S2")).toBeInTheDocument();
    expect(screen.getByText("Nav")).toBeInTheDocument();
    expect(screen.queryByText("Dual Server")).not.toBeInTheDocument();
    expect(screen.queryByText("NavDP Server")).not.toBeInTheDocument();
    expect(screen.queryByText("System2")).not.toBeInTheDocument();
  });
});
