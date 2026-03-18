import type { ArchitectureNode, DashboardState, LogRecord, ProcessRecord } from "./types";

export function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function asArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

export function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

export function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function booleanValue(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

export function formatMs(value: unknown, fallback = "n/a"): string {
  const numeric = numberValue(value);
  return numeric === null ? fallback : `${Math.round(numeric)}ms`;
}

export function formatSeconds(value: unknown, fallback = "n/a"): string {
  const numeric = numberValue(value);
  return numeric === null ? fallback : `${numeric.toFixed(numeric >= 10 ? 0 : 2)}s`;
}

export function formatMeters(value: unknown, fallback = "n/a"): string {
  const numeric = numberValue(value);
  return numeric === null ? fallback : `${numeric.toFixed(2)}m`;
}

export function formatRadians(value: unknown, fallback = "n/a"): string {
  const numeric = numberValue(value);
  return numeric === null ? fallback : `${numeric.toFixed(2)} rad`;
}

export function titleCase(value: string): string {
  if (value.trim() === "") {
    return "unknown";
  }
  return value
    .split(/[_\s-]+/)
    .filter((item) => item !== "")
    .map((item) => item[0].toUpperCase() + item.slice(1))
    .join(" ");
}

export function processByName(state: DashboardState | null, name: string): ProcessRecord | null {
  if (state === null) {
    return null;
  }
  return state.processes.find((item) => item.name === name) ?? null;
}

const EMPTY_NODE: ArchitectureNode = {
  name: "",
  status: "unknown",
  summary: "",
  detail: "",
  required: false,
  metrics: {},
};

export function architectureRoot(state: DashboardState | null): DashboardState["architecture"] | null {
  if (state === null) {
    return null;
  }
  return state.architecture;
}

export function architectureNode(
  state: DashboardState | null,
  name: "gateway" | "mainControlServer",
): ArchitectureNode {
  const root = architectureRoot(state);
  if (root === null) {
    return EMPTY_NODE;
  }
  return root[name];
}

export function architectureModule(
  state: DashboardState | null,
  name: keyof DashboardState["architecture"]["modules"],
): ArchitectureNode {
  const root = architectureRoot(state);
  if (root === null) {
    return EMPTY_NODE;
  }
  return root.modules[name];
}

export function coreNode(
  state: DashboardState | null,
  name: keyof DashboardState["architecture"]["mainControlServer"]["core"],
): ArchitectureNode {
  const root = architectureRoot(state);
  if (root === null) {
    return EMPTY_NODE;
  }
  return root.mainControlServer.core[name];
}

export function recentLogs(state: DashboardState | null, limit = 60): LogRecord[] {
  if (state === null) {
    return [];
  }
  return [...state.logs].slice(-Math.max(limit, 1)).reverse();
}

export function statusTone(status: string): string {
  if (status === "ok" || status === "running" || status === "connected") {
    return "green";
  }
  if (status === "warning" || status === "inactive" || status === "not_required") {
    return "amber";
  }
  if (status === "error" || status === "down" || status === "failed" || status === "exited") {
    return "red";
  }
  return "slate";
}

export function statusLabel(status: string): string {
  if (status === "not_required") {
    return "not required";
  }
  return status.replace(/_/g, " ");
}
