const DEFAULT_PROXY_TARGET = "http://127.0.0.1:8095";
const DEFAULT_BACKEND_HEALTH_TIMEOUT_MS = 2500;
const DEFAULT_BACKEND_HEALTH_CACHE_MS = 1200;
const DEFAULT_BACKEND_HEALTH_GRACE_MS = 5000;

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? "").trim(), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

type FetchLike = (input: string, init?: RequestInit) => Promise<{ ok: boolean }>;

export function resolveProxyTarget(env: NodeJS.ProcessEnv = process.env): string {
  return trimTrailingSlash(String(env.AURA_DASHBOARD_PROXY_TARGET ?? DEFAULT_PROXY_TARGET).trim() || DEFAULT_PROXY_TARGET);
}

export function resolveBackendHealthTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  return parsePositiveInteger(env.AURA_DASHBOARD_BACKEND_HEALTH_TIMEOUT_MS, DEFAULT_BACKEND_HEALTH_TIMEOUT_MS);
}

export function resolveBackendHealthCacheMs(env: NodeJS.ProcessEnv = process.env): number {
  return parsePositiveInteger(env.AURA_DASHBOARD_BACKEND_HEALTH_CACHE_MS, DEFAULT_BACKEND_HEALTH_CACHE_MS);
}

export function resolveBackendHealthGraceMs(env: NodeJS.ProcessEnv = process.env): number {
  return parsePositiveInteger(env.AURA_DASHBOARD_BACKEND_HEALTH_GRACE_MS, DEFAULT_BACKEND_HEALTH_GRACE_MS);
}

export function createBackendHealthProbe(
  proxyTarget: string,
  {
    timeoutMs = DEFAULT_BACKEND_HEALTH_TIMEOUT_MS,
    cacheMs = DEFAULT_BACKEND_HEALTH_CACHE_MS,
    graceMs = DEFAULT_BACKEND_HEALTH_GRACE_MS,
    fetchImpl = fetch as FetchLike,
    now = () => Date.now(),
  }: {
    timeoutMs?: number;
    cacheMs?: number;
    graceMs?: number;
    fetchImpl?: FetchLike;
    now?: () => number;
  } = {},
) {
  let lastCheckedAt = 0;
  let lastHealthy = false;
  let lastHealthyAt: number | null = null;

  return {
    async shouldProxyToBackend(): Promise<boolean> {
      const currentTime = now();
      if (currentTime - lastCheckedAt < cacheMs) {
        return lastHealthy || (lastHealthyAt !== null && currentTime - lastHealthyAt < graceMs);
      }

      lastCheckedAt = currentTime;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetchImpl(`${proxyTarget}/api/bootstrap`, {
          method: "GET",
          signal: controller.signal,
        });
        lastHealthy = response.ok;
        if (response.ok) {
          lastHealthyAt = currentTime;
        }
        return response.ok;
      } catch {
        lastHealthy = false;
        return lastHealthyAt !== null && currentTime - lastHealthyAt < graceMs;
      } finally {
        clearTimeout(timeoutId);
      }
    },
  };
}
