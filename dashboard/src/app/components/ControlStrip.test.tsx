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
    locomotionConfig: {
      actionScale: "0.5",
      onnxDevice: "auto" as const,
      cmdMaxVx: "0.5",
      cmdMaxVy: "0.3",
      cmdMaxWz: "0.8",
    },
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
          locomotionConfig: {
            actionScale: 0.65,
            onnxDevice: "cuda",
            cmdMaxVx: 0.8,
            cmdMaxVy: 0.4,
            cmdMaxWz: 1.0,
          },
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

  it("disables runtime controls while Vite mock mode is active", () => {
    mockContext.form = {
      ...mockContext.form,
      plannerMode: "interactive",
      goalX: "1",
      goalY: "2",
    };
    mockContext.state = {
      ...mockContext.state,
      runtime: {
        lastStatusEvent: {
          state: "mock_mode",
          reason: "AURA dashboard backend is not running at http://127.0.0.1:8095. Vite dev fallback responses are active.",
        },
      },
    };

    render(<ControlStrip />);

    expect(screen.getByRole("button", { name: /start stack/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /stop stack/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /submit task/i })).toBeDisabled();
    expect(screen.getByText(/Vite dev fallback responses are active/i)).toBeInTheDocument();
  });

  it("updates the locomotion config from the dashboard input", () => {
    mockContext.form = {
      ...mockContext.form,
      plannerMode: "interactive",
      locomotionConfig: {
        ...mockContext.form.locomotionConfig,
        actionScale: "0.5",
      },
      goalX: "1",
      goalY: "2",
    };

    render(<ControlStrip />);

    const actionScaleInput = screen
      .getByText("action scale")
      .parentElement?.querySelector("input");
    if (actionScaleInput == null) {
      throw new Error("action scale input not found");
    }
    fireEvent.change(actionScaleInput, {
      target: { value: "0.7" },
    });

    expect(mockContext.setForm).toHaveBeenCalledWith({
      locomotionConfig: {
        ...mockContext.form.locomotionConfig,
        actionScale: "0.7",
      },
    });
  });
});
