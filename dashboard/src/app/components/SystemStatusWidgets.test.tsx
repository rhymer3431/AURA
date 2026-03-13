import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

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
    services: {},
    transport: {},
    logs: [
      { source: "aura_runtime", stream: "event", level: "info", message: "interactive task queued" },
      { source: "dual", stream: "stderr", level: "error", message: "timeout retry" },
    ],
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

vi.mock("../state", () => ({
  useDashboard: () => mockDashboard,
}));

describe("SystemStatusWidgets", () => {
  it("shows not required process state", () => {
    render(<ProcessesWidget />);

    expect(screen.getByText("system2")).toBeInTheDocument();
    expect(screen.getAllByText("not required").length).toBeGreaterThan(0);
  });

  it("renders recent logs from dashboard state", () => {
    render(<LogsWidget />);

    expect(screen.getByText("interactive task queued")).toBeInTheDocument();
    expect(screen.getByText("timeout retry")).toBeInTheDocument();
  });
});
