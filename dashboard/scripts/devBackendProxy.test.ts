import { describe, expect, it, vi } from "vitest";

import {
  createBackendHealthProbe,
  resolveBackendHealthCacheMs,
  resolveBackendHealthGraceMs,
  resolveBackendHealthTimeoutMs,
  resolveProxyTarget,
} from "./devBackendProxy";

describe("devBackendProxy", () => {
  it("uses the explicit proxy target when configured", () => {
    expect(
      resolveProxyTarget({
        AURA_DASHBOARD_PROXY_TARGET: "http://10.0.0.5:8095/",
      } as NodeJS.ProcessEnv),
    ).toBe("http://10.0.0.5:8095");
  });

  it("parses health timing overrides from the environment", () => {
    const env = {
      AURA_DASHBOARD_BACKEND_HEALTH_TIMEOUT_MS: "3200",
      AURA_DASHBOARD_BACKEND_HEALTH_CACHE_MS: "900",
      AURA_DASHBOARD_BACKEND_HEALTH_GRACE_MS: "7000",
    } as NodeJS.ProcessEnv;

    expect(resolveBackendHealthTimeoutMs(env)).toBe(3200);
    expect(resolveBackendHealthCacheMs(env)).toBe(900);
    expect(resolveBackendHealthGraceMs(env)).toBe(7000);
  });

  it("keeps proxying for a short grace window after a transient backend timeout", async () => {
    let currentTime = 0;
    const fetchImpl = vi
      .fn<Parameters<typeof fetch>, ReturnType<typeof fetch>>()
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockRejectedValueOnce(new Error("timeout"));

    const probe = createBackendHealthProbe("http://127.0.0.1:8095", {
      timeoutMs: 50,
      cacheMs: 0,
      graceMs: 5000,
      fetchImpl,
      now: () => currentTime,
    });

    await expect(probe.shouldProxyToBackend()).resolves.toBe(true);
    currentTime = 2500;
    await expect(probe.shouldProxyToBackend()).resolves.toBe(true);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("falls back to mock mode once the grace window expires", async () => {
    let currentTime = 0;
    const fetchImpl = vi
      .fn<Parameters<typeof fetch>, ReturnType<typeof fetch>>()
      .mockResolvedValueOnce(new Response("{}", { status: 200 }))
      .mockRejectedValueOnce(new Error("timeout"));

    const probe = createBackendHealthProbe("http://127.0.0.1:8095", {
      timeoutMs: 50,
      cacheMs: 0,
      graceMs: 1000,
      fetchImpl,
      now: () => currentTime,
    });

    await expect(probe.shouldProxyToBackend()).resolves.toBe(true);
    currentTime = 2500;
    await expect(probe.shouldProxyToBackend()).resolves.toBe(false);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});
