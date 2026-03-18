import { render, screen } from "@testing-library/react";
import { beforeEach, vi } from "vitest";

import { LogsWidget, ProcessesWidget } from "./SystemStatusWidgets";

const mockDashboard = {
  bootstrap: null,
  state: {
    timestamp: 0,
    session: { active: false, startedAt: null, config: null, lastEvent: null },
    processes: [
      {
        name: "system2",
        state: "not_required",
        required: false,
        pid: null,
        exitCode: null,
        startedAt: null,
        healthUrl: "http://127.0.0.1:8080",
        stdoutLog: "tmp/process_logs/dashboard/system2.stdout.log",
        stderrLog: "tmp/process_logs/dashboard/system2.stderr.log",
      },
    ],
    runtime: {},
    sensors: {},
    perception: {},
    memory: {},
    architecture: {
      gateway: { name: "Robot Gateway", status: "ok", summary: "frames live", detail: "", required: true, metrics: {} },
      mainControlServer: {
        name: "Main Control Server",
        status: "ok",
        summary: "idle",
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
        s2: { name: "S2", status: "not_required", summary: "", detail: "", required: false, metrics: {} },
        nav: { name: "Nav", status: "inactive", summary: "", detail: "", required: false, metrics: {} },
        locomotion: { name: "Locomotion", status: "inactive", summary: "", detail: "", required: false, metrics: {} },
        telemetry: { name: "Telemetry", status: "ok", summary: "", detail: "", required: true, metrics: {} },
      },
    },
    services: {},
    transport: {},
    logs: [
      { source: "aura_runtime", stream: "event", level: "info", message: "interactive task queued" },
      { source: "dual", stream: "stderr", level: "error", message: "timeout retry" },
    ],
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

describe("SystemStatusWidgets", () => {
  beforeEach(() => {
    Object.defineProperty(globalThis, "fetch", {
      value: vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ logs: mockDashboard.state.logs }),
      }),
      writable: true,
    });
  });

  it("shows not required process state", () => {
    render(<ProcessesWidget />);

    expect(screen.getByText("system2")).toBeInTheDocument();
    expect(screen.getAllByText("not required").length).toBeGreaterThan(0);
  });

  it("renders recent logs from dashboard state", async () => {
    render(<LogsWidget />);

    expect(await screen.findByText("interactive task queued")).toBeInTheDocument();
    expect(await screen.findByText("timeout retry")).toBeInTheDocument();
  });
});
