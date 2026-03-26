import { describe, expect, it } from "vitest";

import { buildDevApiFallbackResponse, shouldBypassApiProxy } from "./devApiFallback";

describe("devApiFallback", () => {
  it("marks only /api paths for bypass", () => {
    expect(shouldBypassApiProxy("/api/state")).toBe(true);
    expect(shouldBypassApiProxy("/assets/index.css")).toBe(false);
  });

  it("returns bootstrap payload for the dashboard shell", () => {
    const response = buildDevApiFallbackResponse({
      apiBaseUrl: "http://127.0.0.1:8095",
      method: "GET",
      requestUrl: "/api/bootstrap",
    });

    expect(response.status).toBe(200);
    expect(response.headers["Content-Type"]).toContain("application/json");
    expect(JSON.parse(response.body)).toMatchObject({
      apiBaseUrl: "http://127.0.0.1:8095",
      webrtcBasePath: "http://127.0.0.1:8095/api/webrtc",
      plannerModes: ["interactive", "pointgoal"],
    });
  });

  it("returns a persistent SSE response for /api/events", () => {
    const response = buildDevApiFallbackResponse({
      apiBaseUrl: "http://127.0.0.1:8095",
      method: "GET",
      requestUrl: "/api/events",
    });

    expect(response.status).toBe(200);
    expect(response.keepAlive).toBe(true);
    expect(response.headers["Content-Type"]).toContain("text/event-stream");
    expect(response.body).toContain("event: state");
  });

  it("returns structured system2 mock state for /api/state", () => {
    const response = buildDevApiFallbackResponse({
      apiBaseUrl: "http://127.0.0.1:8095",
      method: "GET",
      requestUrl: "/api/state",
    });

    expect(response.status).toBe(200);
    expect(JSON.parse(response.body)).toMatchObject({
      services: {
        system2: {
          name: "system2",
          status: "inactive",
          output: null,
        },
      },
    });
  });

  it("returns a 503 for mutating runtime endpoints while mock mode is active", () => {
    const response = buildDevApiFallbackResponse({
      apiBaseUrl: "http://127.0.0.1:8095",
      method: "POST",
      requestUrl: "/api/session/start",
    });

    expect(response.status).toBe(503);
    expect(JSON.parse(response.body)).toMatchObject({ mock: true });
  });
});
