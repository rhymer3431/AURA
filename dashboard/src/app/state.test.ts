import { buildSessionPayload, dashboardReducer, DEFAULT_FORM } from "./state";

describe("dashboard state helpers", () => {
  it("builds interactive payload without startup pointgoal coordinates", () => {
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
        history: { stale: [], goalDistance: [], navdpLatency: [], dualLatency: [] },
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
    expect(next.history.navdpLatency[next.history.navdpLatency.length - 1]).toEqual({ t: 100, v: 48 });
    expect(next.history.dualLatency[next.history.dualLatency.length - 1]).toEqual({ t: 100, v: 130 });
  });
});
