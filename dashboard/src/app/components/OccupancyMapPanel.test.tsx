import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { OccupancyMapPanel } from "./OccupancyMapPanel";

const mockDashboard = {
  bootstrap: null,
  state: {
    timestamp: 0,
    session: {
      active: true,
      startedAt: 1,
      config: {
        plannerMode: "pointgoal" as const,
        launchMode: "gui" as const,
        scenePreset: "interior agent kujiale 3",
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
        goal: { x: 3.5, y: -1.0 },
      },
      lastEvent: null,
    },
    processes: [],
    runtime: {
      goalDistanceM: 2.4,
      globalRouteEnabled: true,
      globalRouteActive: true,
      globalRouteWaypointIndex: 1,
      globalRouteWaypointCount: 4,
      globalRouteLastReplanReason: "route_missing",
      globalRouteLastError: "",
      globalRouteActiveWaypointXy: [1.0, 0.0],
      globalRouteGoalXy: [3.5, -1.0],
      globalRouteWaypointsWorld: [[1.0, 0.0], [2.0, -0.2], [3.0, -0.6], [3.5, -1.0]],
    },
    sensors: {
      robotPoseXyz: [0.75, -0.1, 0.0],
      robotYawRad: 0.2,
    },
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
    scenePreset: "interior agent kujiale 3",
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
    goalX: "3.5",
    goalY: "-1.0",
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

vi.mock("../network", () => ({
  buildApiUrl: (path: string) => `http://127.0.0.1:8095${path}`,
  requestJson: vi.fn().mockResolvedValue({
    available: true,
    scenePreset: "interior agent kujiale 3",
    canonicalScenePreset: "interior agent kujiale 3",
    label: "Interior Agent Kujiale 3",
    imagePath: "/api/occupancy/image?scenePreset=interior+agent+kujiale+3",
    imageWidth: 328,
    imageHeight: 281,
    xMin: -7.825,
    xMax: 8.525,
    yMin: -7.025,
    yMax: 6.975,
    resolutionMpp: 0.05,
  }),
}));

describe("OccupancyMapPanel", () => {
  it("renders occupancy metadata and live route state", async () => {
    render(<OccupancyMapPanel />);

    expect(screen.getByRole("heading", { name: "Occupancy Map" })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByAltText("Occupancy map for Interior Agent Kujiale 3")).toBeInTheDocument();
    });

    expect(screen.getByText("route_missing")).toBeInTheDocument();
    expect(screen.getByText("0.75, -0.10")).toBeInTheDocument();
    expect(screen.getByText("1.00, 0.00")).toBeInTheDocument();
  });
});
