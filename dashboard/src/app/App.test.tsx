import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, vi } from "vitest";

import App from "./App";

const mockDashboard = {
  bootstrap: {
    plannerModes: ["interactive", "pointgoal"],
    launchModes: ["gui", "headless"],
    scenePresets: ["warehouse"],
    apiBaseUrl: "http://127.0.0.1:8095",
    devOrigin: "",
    webrtcBasePath: "http://127.0.0.1:8095/api/webrtc",
  },
  state: {
    timestamp: 0,
    session: {
      active: false,
      startedAt: null,
      config: null,
      lastEvent: null,
    },
    processes: [],
    runtime: {},
    sensors: {},
    perception: {},
    memory: {},
    services: {},
    transport: {},
    logs: [],
  },
  history: { stale: [], goalDistance: [], navdpLatency: [], dualLatency: [] },
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

vi.mock("./state", () => ({
  useDashboard: () => mockDashboard,
}));

vi.mock("./components/OperationsPage", () => ({
  OperationsPage: () => <div>OperationsPage</div>,
}));

vi.mock("./components/NavigationPage", () => ({
  NavigationPage: () => <div>NavigationPage</div>,
}));

vi.mock("./components/DiagnosticsPage", () => ({
  DiagnosticsPage: () => <div>DiagnosticsPage</div>,
}));

vi.mock("./components/SessionConfigPage", () => ({
  SessionConfigPage: () => <div>SessionConfigPage</div>,
}));

vi.mock("./components/SystemStatusWidgets", () => ({
  PerceptionWidget: () => <div>PerceptionWidget</div>,
  MemoryWidget: () => <div>MemoryWidget</div>,
}));

describe("App navigation", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("renders overview by default and switches pages from the sidebar", async () => {
    render(<App />);

    expect(screen.getByText("OperationsPage")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Operations" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Navigation" }));

    await waitFor(() => {
      expect(screen.getByText("NavigationPage")).toBeInTheDocument();
      expect(screen.queryByText("OperationsPage")).not.toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Navigation" })).toBeInTheDocument();
      expect(window.location.hash).toBe("#/navigation");
    });
  });
});
