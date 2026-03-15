import { buildSessionPayload, dashboardReducer, DEFAULT_FORM } from "./state";

describe("dashboard state helpers", () => {
  it("builds pointgoal payload with numeric coordinates", () => {
    const payload = buildSessionPayload({
      ...DEFAULT_FORM,
      plannerMode: "pointgoal",
      policyPath: "  artifacts/models/policy.onnx  ",
      goalX: "1.5",
      goalY: "-2.25",
    });

    expect(payload.goal).toEqual({ x: 1.5, y: -2.25 });
    expect(payload.policyPath).toBe("artifacts/models/policy.onnx");
  });

  it("rejects invalid pointgoal coordinates", () => {
    expect(() =>
      buildSessionPayload({
        ...DEFAULT_FORM,
        plannerMode: "pointgoal",
        goalX: "abc",
        goalY: "0",
      }),
    ).toThrow("pointgoal goal must contain numeric x and y values");
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
