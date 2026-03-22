import { RefreshCw } from "lucide-react";

import type { DashboardPage } from "../navigation";
import { useDashboard } from "../state";
import { architectureNode, statusLabel, stringValue } from "../selectors";
import { ConsoleBadge, ConsolePanel } from "./console-ui";

function formatStartedAt(value: number | null): string {
  if (value === null) {
    return "not running";
  }

  const timestampMs = value < 1_000_000_000_000 ? value * 1000 : value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(timestampMs);
}

export function DashboardPageHero({ page }: { page: DashboardPage }) {
  const { refresh, state } = useDashboard();
  const Icon = page.icon;
  const gateway = architectureNode(state, "gateway");
  const controlServer = architectureNode(state, "mainControlServer");
  const runtimeMode = String((state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE"));
  const sessionConfig = state?.session.config;
  const sessionTone = state?.session.active ? "emerald" : "amber";
  const sessionLabel = state?.session.active ? "runtime live" : "runtime idle";

  const fields = [
    {
      label: "Session profile",
      value: `${sessionConfig?.scenePreset ?? "draft"} / ${sessionConfig?.launchMode ?? "gui"}`,
      note: `viewer ${sessionConfig?.viewerEnabled ? "on" : "off"} · detection ${sessionConfig?.detectionEnabled ? "on" : "off"}`,
    },
    {
      label: "Execution mode",
      value: runtimeMode,
      note: `started ${formatStartedAt(state?.session.startedAt ?? null)}`,
    },
    {
      label: "Gateway",
      value: gateway.summary || statusLabel(gateway.status),
      note: gateway.detail || gateway.name || "robot ingress mirror",
    },
    {
      label: "Main Control Server",
      value: controlServer.summary || statusLabel(controlServer.status),
      note: stringValue(controlServer.detail, "task arbitration rail"),
    },
  ];

  return (
    <ConsolePanel className="dashboard-page-hero">
      <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-4">
          <div className="dashboard-page-hero-icon">
            <Icon className="size-6" />
          </div>
          <div className="min-w-0">
            <div className="dashboard-eyebrow mb-2">Workspace Context</div>
            <div className="text-[22px] font-medium tracking-[-0.05em] text-[var(--foreground)]">{page.label}</div>
            <p className="dashboard-subtitle mt-2 max-w-2xl">{page.description}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className="dashboard-button-secondary" onClick={() => void refresh()}>
            <RefreshCw className="size-4" />
            Refresh Snapshot
          </button>
          <ConsoleBadge tone={sessionTone}>{sessionLabel}</ConsoleBadge>
        </div>
      </div>

      <div className="dashboard-page-hero-grid mt-5">
        {fields.map((field) => (
          <div key={field.label} className="dashboard-summary-field">
            <div className="dashboard-eyebrow mb-2">{field.label}</div>
            <div className="text-[14px] font-medium tracking-[-0.03em] text-[var(--foreground)]">{field.value}</div>
            <div className="dashboard-subtitle mt-2">{field.note}</div>
          </div>
        ))}
      </div>
    </ConsolePanel>
  );
}
