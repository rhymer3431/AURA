import {
  Activity,
  Bell,
  Bot,
  Database,
  Radio,
  Scan,
} from "lucide-react";

import { useDashboard } from "../state";
import {
  architectureModule,
  architectureNode,
  recentLogs,
  statusLabel,
  statusTone,
  stringValue,
} from "../selectors";
import { ConsoleBadge, ConsolePanel, toneFromStatusTone } from "./console-ui";

function relativeTime(timestampNs?: number): string {
  if (typeof timestampNs !== "number" || !Number.isFinite(timestampNs)) {
    return "live";
  }

  const deltaMs = Date.now() - timestampNs / 1_000_000;
  if (deltaMs < 60_000) {
    return "just now";
  }
  if (deltaMs < 3_600_000) {
    return `${Math.max(1, Math.round(deltaMs / 60_000))}m ago`;
  }
  if (deltaMs < 86_400_000) {
    return `${Math.max(1, Math.round(deltaMs / 3_600_000))}h ago`;
  }
  return `${Math.max(1, Math.round(deltaMs / 86_400_000))}d ago`;
}

function RailItem({
  icon: Icon,
  title,
  detail,
  meta,
  badge,
}: {
  icon: typeof Bell;
  title: string;
  detail: string;
  meta: string;
  badge?: React.ReactNode;
}) {
  return (
    <div className="dashboard-activity-item">
      <div className="dashboard-activity-icon">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-[12px] font-medium text-[var(--foreground)]">{title}</div>
            <div className="dashboard-subtitle mt-1 line-clamp-2">{detail}</div>
          </div>
          {badge}
        </div>
        <div className="dashboard-activity-meta mt-2">{meta}</div>
      </div>
    </div>
  );
}

export function DashboardActivityRail() {
  const { state } = useDashboard();
  const runtimeMode = String((state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE"));
  const scenePreset = state?.session.config?.scenePreset ?? "draft";
  const lastEvent = state?.session.lastEvent;
  const logs = recentLogs(state, 4);
  const actors = [
    {
      icon: Radio,
      name: architectureNode(state, "gateway").name || "Gateway",
      summary: architectureNode(state, "gateway").summary || architectureNode(state, "gateway").detail || "idle",
      status: architectureNode(state, "gateway").status,
    },
    {
      icon: Bot,
      name: architectureNode(state, "mainControlServer").name || "Main Control Server",
      summary: architectureNode(state, "mainControlServer").summary || architectureNode(state, "mainControlServer").detail || "idle",
      status: architectureNode(state, "mainControlServer").status,
    },
    {
      icon: Scan,
      name: architectureModule(state, "perception").name || "Perception",
      summary: architectureModule(state, "perception").summary || architectureModule(state, "perception").detail || "idle",
      status: architectureModule(state, "perception").status,
    },
    {
      icon: Database,
      name: architectureModule(state, "memory").name || "Memory",
      summary: architectureModule(state, "memory").summary || architectureModule(state, "memory").detail || "idle",
      status: architectureModule(state, "memory").status,
    },
  ];

  return (
    <aside className="dashboard-activity-rail">
      <ConsolePanel className="dashboard-rail-block">
        <div className="dashboard-eyebrow mb-3">Notifications</div>
        <div className="space-y-3">
          <RailItem
            icon={Bell}
            title={state?.session.active ? "Runtime session is live." : "Runtime session is idle."}
            detail={`scene ${scenePreset} · mode ${runtimeMode}`}
            meta={lastEvent?.timestampNs ? relativeTime(lastEvent.timestampNs) : "session watcher"}
            badge={<ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>{state?.session.active ? "live" : "idle"}</ConsoleBadge>}
          />
          <RailItem
            icon={Activity}
            title={lastEvent?.message ?? "No recent task event"}
            detail={lastEvent?.details ? JSON.stringify(lastEvent.details).slice(0, 80) : "Waiting for the next runtime transition."}
            meta={`${lastEvent?.source ?? "main-control-server"} · ${lastEvent?.stream ?? "events"}`}
          />
        </div>
      </ConsolePanel>

      <ConsolePanel className="dashboard-rail-block">
        <div className="dashboard-eyebrow mb-3">Activity</div>
        <div className="space-y-3">
          {logs.length > 0 ? logs.map((log, index) => (
            <RailItem
              key={`${log.source}-${log.stream}-${index}`}
              icon={Activity}
              title={log.message}
              detail={`${log.source} · ${log.stream}`}
              meta={relativeTime(log.timestampNs)}
            />
          )) : (
            <div className="dashboard-subtitle">No recent logs streamed into the dashboard yet.</div>
          )}
        </div>
      </ConsolePanel>

      <ConsolePanel className="dashboard-rail-block">
        <div className="dashboard-eyebrow mb-3">Runtime Actors</div>
        <div className="space-y-3">
          {actors.map((actor) => (
            <RailItem
              key={actor.name}
              icon={actor.icon}
              title={actor.name}
              detail={stringValue(actor.summary, "idle")}
              meta={statusLabel(actor.status)}
              badge={<ConsoleBadge tone={toneFromStatusTone(statusTone(actor.status))}>{statusLabel(actor.status)}</ConsoleBadge>}
            />
          ))}
        </div>
      </ConsolePanel>
    </aside>
  );
}
