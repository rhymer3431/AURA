const DEFAULT_DESKTOP_API_BASE = "http://127.0.0.1:8095";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function resolveApiBaseUrl(): string {
  const configuredBase = String(import.meta.env.VITE_AURA_API_BASE ?? "").trim();
  if (configuredBase !== "") {
    return trimTrailingSlash(configuredBase);
  }
  if (import.meta.env.DEV) {
    return "";
  }
  return DEFAULT_DESKTOP_API_BASE;
}

export function buildApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = resolveApiBaseUrl();
  return `${base}${normalizedPath}`;
}

async function readErrorMessage(response: Response): Promise<string> {
  const bodyText = (await response.text()).trim();
  if (bodyText === "") {
    return `${response.status} ${response.statusText}`.trim();
  }
  try {
    const parsed = JSON.parse(bodyText) as { error?: unknown; message?: unknown };
    if (typeof parsed.error === "string" && parsed.error.trim() !== "") {
      return parsed.error;
    }
    if (typeof parsed.message === "string" && parsed.message.trim() !== "") {
      return parsed.message;
    }
  } catch {
    // Non-JSON error bodies should be surfaced as-is.
  }
  return bodyText;
}

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildApiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return (await response.json()) as T;
}

export function createDashboardEventSource(path: string): EventSource {
  return new EventSource(buildApiUrl(path));
}
