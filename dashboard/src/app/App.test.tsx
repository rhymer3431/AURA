import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, vi } from "vitest";

import App from "./App";

const mockDashboard = {
  bootstrap: {
    plannerModes: ["interactive"],
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

vi.mock("./components/StatCards", () => ({
  StatCards: () => <div>StatCards</div>,
}));

vi.mock("./components/PipelineFlow", () => ({
  PipelineFlow: () => <div>PipelineFlow</div>,
}));

vi.mock("./components/NavigationControlPanel", () => ({
  NavigationControlPanel: () => <div>NavigationControlPanel</div>,
}));

vi.mock("./components/ExternalServicesPanel", () => ({
  ExternalServicesPanel: () => <div>ExternalServicesPanel</div>,
}));

vi.mock("./components/RobotViewer", () => ({
  RobotViewer: () => <div>RobotViewer</div>,
}));

vi.mock("./components/ControlStrip", () => ({
  ControlStrip: () => <div>ControlStrip</div>,
}));

vi.mock("./components/SystemStatusWidgets", () => ({
  ProcessesWidget: () => <div>ProcessesWidget</div>,
  SensorsWidget: () => <div>SensorsWidget</div>,
  PerceptionWidget: () => <div>PerceptionWidget</div>,
  MemoryWidget: () => <div>MemoryWidget</div>,
  IpcOrchestrationWidget: () => <div>IpcOrchestrationWidget</div>,
  LogsWidget: () => <div>LogsWidget</div>,
}));

vi.mock("./components/ExecutionModesPanel", () => ({
  ExecutionModesPanel: () => <div>ExecutionModesPanel</div>,
}));

vi.mock("./components/ArtifactsStoragePanel", () => ({
  ArtifactsStoragePanel: () => <div>ArtifactsStoragePanel</div>,
}));

describe("App navigation", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("renders overview by default and switches pages from the sidebar", async () => {
    render(<App />);

    expect(screen.getByText("StatCards")).toBeInTheDocument();
    expect(screen.getByText("PipelineFlow")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Pipeline Overview" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Planner & Control" }));

    await waitFor(() => {
      expect(screen.getByText("NavigationControlPanel")).toBeInTheDocument();
      expect(screen.queryByText("PipelineFlow")).not.toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Planner & Control" })).toBeInTheDocument();
      expect(window.location.hash).toBe("#/planner-control");
    });
  });
});
