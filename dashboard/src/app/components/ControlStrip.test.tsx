import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { ControlStrip } from "./ControlStrip";

const mockContext: any = {
  bootstrap: {
    plannerModes: ["interactive", "pointgoal"],
    launchModes: ["gui", "headless"],
    scenePresets: ["warehouse", "interior agent kujiale 3"],
    apiBaseUrl: "http://127.0.0.1:8095",
    devOrigin: "",
    webrtcBasePath: "http://127.0.0.1:8095/api/webrtc",
  },
  state: {
    timestamp: 0,
    session: { active: false, startedAt: null, config: null, lastEvent: null },
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
    plannerMode: "pointgoal" as const,
    launchMode: "gui" as const,
    scenePreset: "warehouse",
    viewerEnabled: true,
    memoryStore: true,
    detectionEnabled: true,
    goalX: "abc",
    goalY: "0",
  },
  loading: false,
  error: "",
  setForm: vi.fn(),
  startSession: vi.fn().mockResolvedValue(undefined),
  stopSession: vi.fn().mockResolvedValue(undefined),
  submitTask: vi.fn().mockResolvedValue(undefined),
  cancelTask: vi.fn().mockResolvedValue(undefined),
  refresh: vi.fn().mockResolvedValue(undefined),
};

vi.mock("../state", () => ({
  useDashboard: () => mockContext,
}));

describe("ControlStrip", () => {
  it("disables session start when pointgoal coordinates are invalid", () => {
    render(<ControlStrip />);

    expect(screen.getByText("pointgoal 모드에서는 numeric `goal x / goal y`가 필요합니다.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start stack/i })).toBeDisabled();
  });

  it("submits interactive task only for active interactive session", () => {
    mockContext.form = {
      ...mockContext.form,
      plannerMode: "interactive",
      goalX: "1",
      goalY: "2",
    };
    mockContext.state = {
      ...mockContext.state,
      session: {
        active: true,
        startedAt: 1,
        config: {
          plannerMode: "interactive",
          launchMode: "gui",
          scenePreset: "warehouse",
          viewerEnabled: true,
          memoryStore: true,
          detectionEnabled: true,
        },
        lastEvent: null,
      },
    };

    render(<ControlStrip />);
    fireEvent.change(screen.getByPlaceholderText("자연어 task를 입력하세요"), {
      target: { value: "go to the loading dock" },
    });
    fireEvent.click(screen.getByRole("button", { name: /submit task/i }));

    expect(mockContext.submitTask).toHaveBeenCalledWith("go to the loading dock");
  });

  it("renders and selects the kujiale 3 scene preset", () => {
    mockContext.form = {
      ...mockContext.form,
      plannerMode: "interactive",
      scenePreset: "warehouse",
      goalX: "1",
      goalY: "2",
    };

    render(<ControlStrip />);

    expect(screen.getByRole("option", { name: "interior agent kujiale 3" })).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue("warehouse"), {
      target: { value: "interior agent kujiale 3" },
    });

    expect(mockContext.setForm).toHaveBeenCalledWith({ scenePreset: "interior agent kujiale 3" });
  });
});
