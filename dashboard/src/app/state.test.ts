import { buildSessionPayload, dashboardReducer, DEFAULT_FORM } from "./state";

describe("dashboard state helpers", () => {
  it("builds session payload with locomotion config only", () => {
    const payload = buildSessionPayload({
      ...DEFAULT_FORM,
      locomotionConfig: {
        ...DEFAULT_FORM.locomotionConfig,
        actionScale: "0.65",
        onnxDevice: "cuda",
      },
    });

    expect(payload).not.toHaveProperty("goal");
    expect(payload.locomotionConfig).toEqual({
      actionScale: 0.65,
      onnxDevice: "cuda",
      cmdMaxVx: 0.5,
      cmdMaxVy: 0.3,
      cmdMaxWz: 0.8,
    });
  });

  it("rejects invalid locomotion config values", () => {
    expect(() =>
      buildSessionPayload({
        ...DEFAULT_FORM,
        locomotionConfig: {
          ...DEFAULT_FORM.locomotionConfig,
          actionScale: "abc",
        },
      }),
    ).toThrow("locomotion config must contain numeric values");
  });

  it("hydrates state history from runtime and service snapshots", () => {
    const next = dashboardReducer(
      {
        bootstrap: null,
        state: null,
        history: { stale: [], goalDistance: [], navLatency: [], s2Latency: [] },
        error: "",
      },
      {
        type: "state",
        payload: {
          timestamp: 100,
          session: { active: true, startedAt: 100, config: null, lastEvent: null },
          processes: [],
          runtime: { staleSec: 0.4, goalDistanceM: 2.3 },
          sensors: {},
          perception: {},
          memory: {},
          architecture: {
            gateway: { name: "Robot Gateway", status: "ok", summary: "", detail: "", required: true, metrics: {} },
            mainControlServer: {
              name: "Main Control Server",
              status: "ok",
              summary: "",
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
              s2: { name: "S2", status: "ok", summary: "", detail: "", required: true, latencyMs: 130, metrics: {} },
              nav: { name: "Nav", status: "ok", summary: "", detail: "", required: true, latencyMs: 48, metrics: {} },
              locomotion: { name: "Locomotion", status: "ok", summary: "", detail: "", required: true, metrics: {} },
              telemetry: { name: "Telemetry", status: "ok", summary: "", detail: "", required: true, metrics: {} },
            },
          },
          services: {
            navdp: { name: "navdp", status: "ok", latencyMs: 48 },
            dual: { name: "dual", status: "ok", latencyMs: 130 },
          },
          transport: {},
          logs: [],
        },
      },
    );

    expect(next.history.stale[next.history.stale.length - 1]).toEqual({ t: 100, v: 0.4 });
    expect(next.history.goalDistance[next.history.goalDistance.length - 1]).toEqual({ t: 100, v: 2.3 });
    expect(next.history.navLatency[next.history.navLatency.length - 1]).toEqual({ t: 100, v: 48 });
    expect(next.history.s2Latency[next.history.s2Latency.length - 1]).toEqual({ t: 100, v: 130 });
  });
});
