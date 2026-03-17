import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { PipelineFlow } from "./PipelineFlow";

const mockDashboard = {
  state: {
    timestamp: 0,
    session: {
      active: true,
      startedAt: 1,
      config: {
        plannerMode: "interactive" as const,
        launchMode: "gui" as const,
        scenePreset: "warehouse",
        viewerEnabled: true,
        memoryStore: true,
        detectionEnabled: true,
        locomotionConfig: {
          actionScale: 0.5,
          onnxDevice: "auto" as const,
          cmdMaxVx: 0.5,
          cmdMaxVy: 0.3,
          cmdMaxWz: 0.8,
        },
      },
      lastEvent: null,
    },
    processes: [],
    runtime: {
      ownerComponent: "navigation_runtime",
      ownerDisplayName: "NavigationRuntime",
      plannerControlMode: "interactive",
      activeCommandType: "LOCAL_SEARCH",
      trajVersion: 3,
    },
    sensors: { rgbAvailable: true, source: "navigation_runtime" },
    perception: { detectorBackend: "stub", detectorReady: true },
    memory: { objectCount: 2, scratchpad: { taskState: "active", nextPriority: "inspect target shelf" } },
    services: {
      navdp: { status: "ok", latencyMs: 12 },
      dual: { status: "not_required" },
    },
    transport: { viewerEnabled: true, frameAgeMs: 18, frameAvailable: true, peerActive: false },
    logs: [],
  },
};

vi.mock("../state", () => ({
  useDashboard: () => mockDashboard,
}));

describe("PipelineFlow", () => {
  it("renders the canonical NavigationRuntime module flow", () => {
    render(<PipelineFlow />);

    expect(screen.getByText(/Navigation\s+Runtime/)).toBeInTheDocument();
    expect(screen.getByText(/Observation\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/World Model\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/Mission\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/Planning\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/Execution\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/Runtime I\/O\s+Module/)).toBeInTheDocument();
    expect(screen.getByText(/locomotion\.\s*runtime/)).toBeInTheDocument();
    expect(screen.getByText(/owner: NavigationRuntime/)).toBeInTheDocument();
  });
});
