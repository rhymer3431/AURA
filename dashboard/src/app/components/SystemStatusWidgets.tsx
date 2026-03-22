import { useEffect, useState } from "react";
import {
  Scan,
  Database,
  Eye,
  Layers,
  Box,
  MonitorSmartphone,
  Terminal,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

import { useDashboard } from "../state";
import { requestJson } from "../network";
import type { LogRecord } from "../types";
import {
  architectureModule,
  architectureNode,
  coreNode,
  asRecord,
  booleanValue,
  formatMs,
  recentLogs,
  statusLabel,
  statusTone,
  stringValue,
} from "../selectors";
import { ConsoleBadge, ConsoleInfoRow, ConsolePanel, ConsoleSectionTitle, toneFromStatusTone } from "./console-ui";

function SectionHeader({ icon: Icon, title }: { icon: typeof Scan; title: string }) {
  return (
    <div className="mb-2.5 flex items-center gap-2">
      <div className="dashboard-icon-shell !size-8">
        <Icon className="size-4" />
      </div>
      <span className="text-[13px] font-medium text-[var(--foreground)]">{title}</span>
    </div>
  );
}

function Chip({
  color,
  children,
}: {
  color: "green" | "blue" | "amber" | "violet" | "slate";
  children: React.ReactNode;
}) {
  const tones = {
    green: "emerald",
    blue: "cyan",
    amber: "amber",
    violet: "violet",
    slate: "slate",
  };
  return <ConsoleBadge tone={tones[color] as "emerald" | "cyan" | "amber" | "violet" | "slate"}>{children}</ConsoleBadge>;
}

function toneForStatus(status: string): "green" | "blue" | "amber" | "violet" | "slate" {
  if (statusTone(status) === "green") {
    return "green";
  }
  if (statusTone(status) === "amber") {
    return "amber";
  }
  if (statusTone(status) === "red") {
    return "violet";
  }
  return "slate";
}

export function ProcessesWidget() {
  const { state } = useDashboard();
  const processes = state?.processes ?? [];

  return (
    <ConsolePanel>
      <ConsoleSectionTitle
        icon={Box}
        eyebrow="runtime layout"
        title="프로세스 구성"
        description="runtime process mirror, pid, and supervisor state"
        className="mb-3.5"
      />
      <div className="space-y-2">
        {processes.map((process) => (
          <div
            key={process.name}
            className="dashboard-panel-strong flex items-center gap-3 px-3.5 py-2.5 transition-all"
          >
            <span
              className={`size-2 rounded-full ${
                statusTone(process.state) === "green"
                  ? "bg-emerald-500"
                  : statusTone(process.state) === "amber"
                    ? "bg-amber-500"
                    : statusTone(process.state) === "red"
                      ? "bg-red-500"
                      : "bg-slate-400"
              }`}
            />
            <div className="flex-1 min-w-0">
              <div className="text-[12px] text-[var(--foreground)] truncate">{process.name}</div>
              <div className="dashboard-micro">
                {statusLabel(process.state)} · PID {process.pid ?? "n/a"}
              </div>
            </div>
            <Chip color={toneForStatus(process.state)}>{statusLabel(process.state)}</Chip>
          </div>
        ))}
      </div>
    </ConsolePanel>
  );
}

export function SensorsWidget() {
  const { state } = useDashboard();
  const sensors = asRecord(state?.sensors);
  const gateway = architectureNode(state, "gateway");

  return (
    <ConsolePanel>
      <div className="flex items-center justify-between mb-3">
        <SectionHeader icon={MonitorSmartphone} title="센서 입력 상태" />
        <Chip color={toneForStatus(gateway.status)}>{statusLabel(gateway.status)}</Chip>
      </div>
      <div className="mb-3.5 grid grid-cols-3 gap-2.5">
        {[
          { label: "RGB", ok: booleanValue(sensors.rgbAvailable) },
          { label: "Depth", ok: booleanValue(sensors.depthAvailable) },
          { label: "Pose", ok: booleanValue(sensors.poseAvailable) },
        ].map((sensor) => (
          <div key={sensor.label} className="dashboard-panel-strong py-2.5 text-center">
            <span className={`inline-block size-2.5 rounded-full ${sensor.ok ? "bg-emerald-500" : "bg-amber-500"} mb-1.5`} />
            <div className="text-[12px] font-medium text-[var(--text-secondary)]">{sensor.label}</div>
          </div>
        ))}
      </div>
      <div className="space-y-1.5 text-[11px]">
        <ConsoleInfoRow label="Gateway" value={<span className="truncate rounded-full bg-[rgba(var(--ink-rgb),0.05)] px-2 py-0.5">{gateway.summary || "idle"}</span>} />
        <ConsoleInfoRow label="Frame Freshness" value={formatMs(state?.transport.frameAgeMs, "n/a")} valueClassName="text-[var(--signal-emerald)] font-medium" />
        <ConsoleInfoRow
          label="Sensor Init"
          value={booleanValue(sensors.rgbAvailable) && booleanValue(sensors.depthAvailable) && booleanValue(sensors.poseAvailable) ? "ready" : "pending"}
          valueClassName="font-medium text-[var(--foreground)]"
        />
        <ConsoleInfoRow label="Frame Source" value={<span className="truncate rounded-full bg-[rgba(var(--ink-rgb),0.05)] px-2 py-0.5">{stringValue(sensors.source, "n/a")}</span>} />
        <ConsoleInfoRow label="Frame ID" value={String(sensors.frameId ?? "n/a")} />
      </div>
    </ConsolePanel>
  );
}

export function PerceptionWidget() {
  const { state } = useDashboard();
  const perception = asRecord(state?.perception);
  const capability = asRecord(perception.detectorCapability);
  const module = architectureModule(state, "perception");

  return (
    <ConsolePanel>
      <div className="mb-2.5 flex items-center justify-between">
        <SectionHeader icon={Scan} title="Perception Module" />
        <Chip color={toneForStatus(module.status)}>{statusLabel(module.status)}</Chip>
      </div>
      <div className="mb-2.5 flex items-center justify-between text-[11px]">
        <span className="dashboard-meta">{module.summary || "Detector Backend"}</span>
        <Chip color={booleanValue(perception.detectorReady) ? "green" : "amber"}>
          {stringValue(perception.detectorBackend, stringValue(capability.backend_name, "unknown"))}
        </Chip>
      </div>
      <div className="mb-3.5 grid grid-cols-2 gap-2.5">
        {[
          { label: "Detection", value: String(perception.detectionCount ?? 0) },
          { label: "Tracked", value: String(perception.trackedDetectionCount ?? 0) },
          { label: "Trajectory", value: String(perception.trajectoryPointCount ?? 0) },
          { label: "Ready", value: booleanValue(perception.detectorReady) ? "yes" : "no" },
        ].map((card) => (
          <div key={card.label} className="dashboard-panel-strong px-3 py-2.5">
            <div className="dashboard-eyebrow mb-1">{card.label}</div>
            <div className="dashboard-mono text-[16px] font-semibold text-[var(--foreground)]">{card.value}</div>
          </div>
        ))}
      </div>
      <div className="space-y-1.5 text-[11px]">
        <ConsoleInfoRow label="Module Detail" value={<span className="rounded-full bg-[rgba(var(--ink-rgb),0.05)] px-2 py-0.5">{module.detail || "n/a"}</span>} />
        <ConsoleInfoRow label="Selected Reason" value={<span className="rounded-full bg-[rgba(var(--ink-rgb),0.05)] px-2 py-0.5">{stringValue(perception.detectorSelectedReason, "n/a")}</span>} />
        <div className="flex justify-between items-center"><span className="dashboard-meta">Capability Status</span><Chip color={stringValue(capability.status) === "ready" ? "green" : "amber"}>{stringValue(capability.status, "unknown")}</Chip></div>
      </div>
    </ConsolePanel>
  );
}

export function MemoryWidget() {
  const { state } = useDashboard();
  const memory = asRecord(state?.memory);
  const scratchpad = asRecord(memory.scratchpad);
  const module = architectureModule(state, "memory");

  return (
    <ConsolePanel className="h-full flex flex-col">
      <div className="mb-2.5 flex items-center justify-between">
        <SectionHeader icon={Database} title="Memory Module" />
        <Chip color={toneForStatus(module.status)}>{statusLabel(module.status)}</Chip>
      </div>
      <div className="mb-2.5 space-y-2 text-[11px]">
        <div className="flex justify-between items-center"><span className="dashboard-meta">Module Summary</span><Chip color={toneForStatus(module.status)}>{module.summary || "idle"}</Chip></div>
        <div className="flex justify-between items-center"><span className="dashboard-meta">Memory-Aware</span><Chip color={booleanValue(memory.memoryAwareTaskActive) ? "blue" : "slate"}>{booleanValue(memory.memoryAwareTaskActive) ? "task_active" : "idle"}</Chip></div>
        <div className="flex justify-between items-center"><span className="dashboard-meta">Scratchpad</span><Chip color={stringValue(scratchpad.taskState) === "active" ? "green" : "amber"}>{stringValue(scratchpad.taskState, "idle")}</Chip></div>
      </div>
      <div className="mb-3.5 grid grid-cols-3 gap-2.5">
        {[
          { label: "Objects", value: String(memory.objectCount ?? 0) },
          { label: "Places", value: String(memory.placeCount ?? 0) },
          { label: "Rules", value: String(memory.semanticRuleCount ?? 0) },
        ].map((card) => (
          <div key={card.label} className="dashboard-panel-strong px-3 py-2.5 text-center">
            <div className="dashboard-eyebrow mb-1">{card.label}</div>
            <div className="dashboard-mono text-[16px] font-semibold text-[var(--foreground)]">{card.value}</div>
          </div>
        ))}
      </div>
      <div className="dashboard-panel-strong flex-1 p-3">
        <div className="dashboard-eyebrow mb-2">Scratchpad (Current Task)</div>
        <div className="dashboard-field dashboard-mono text-[12px] text-[var(--text-secondary)] leading-tight break-words">
          {stringValue(scratchpad.instruction, "idle")}
        </div>
      </div>
      <div className="mt-2 rounded-full border border-[var(--tone-cyan-border)] bg-[var(--tone-cyan-bg)] px-2 py-1.5 text-[10px] text-[var(--tone-cyan-fg)] text-center dashboard-mono">
        next priority: {stringValue(scratchpad.nextPriority, "n/a")}
      </div>
    </ConsolePanel>
  );
}

export function MainControlServerWidget() {
  const { state } = useDashboard();
  const server = architectureNode(state, "mainControlServer");
  const coreEntries = [
    coreNode(state, "worldStateStore"),
    coreNode(state, "decisionEngine"),
    coreNode(state, "plannerCoordinator"),
    coreNode(state, "commandResolver"),
    coreNode(state, "safetySupervisor"),
  ];
  const metrics = asRecord(server.metrics);

  return (
    <ConsolePanel>
      <div className="mb-3.5 flex items-center justify-between">
        <SectionHeader icon={Layers} title="Main Control Server" />
        <Chip color={toneForStatus(server.status)}>{statusLabel(server.status)}</Chip>
      </div>
      <div className="dashboard-panel-strong mb-3.5 px-3.5 py-3">
        <div className="text-[12px] font-medium text-[var(--foreground)]">{server.summary || "Ready"}</div>
        <div className="dashboard-micro mt-1">{server.detail || "No active runtime task"}</div>
      </div>
      <div className="mb-3.5 grid grid-cols-2 gap-2.5 text-[11px]">
        <div className="dashboard-panel-strong px-3 py-2.5">
          <div className="dashboard-eyebrow mb-1">Mode</div>
          <div className="font-medium text-[var(--foreground)]">{stringValue(metrics.mode, "idle")}</div>
        </div>
        <div className="dashboard-panel-strong px-3 py-2.5">
          <div className="dashboard-eyebrow mb-1">Task State</div>
          <div className="font-medium text-[var(--foreground)]">{stringValue(metrics.taskState, "idle")}</div>
        </div>
        <div className="dashboard-panel-strong px-3 py-2.5">
          <div className="dashboard-eyebrow mb-1">Control Mode</div>
          <div className="font-medium text-[var(--foreground)]">{stringValue(metrics.plannerControlMode, "idle")}</div>
        </div>
        <div className="dashboard-panel-strong px-3 py-2.5">
          <div className="dashboard-eyebrow mb-1">Recovery</div>
          <div className="font-medium text-[var(--foreground)]">{stringValue(metrics.recoveryState, "NORMAL")}</div>
        </div>
      </div>
      <div className="space-y-2">
        {coreEntries.map((node) => (
          <div key={node.name} className="dashboard-panel-strong px-3.5 py-2.5">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[12px] font-medium text-[var(--foreground)]">{node.name}</div>
                <div className="dashboard-micro mt-1 truncate">{node.summary || node.detail || "idle"}</div>
              </div>
              <Chip color={toneForStatus(node.status)}>{statusLabel(node.status)}</Chip>
            </div>
          </div>
        ))}
      </div>
    </ConsolePanel>
  );
}

export function IpcOrchestrationWidget() {
  const { state } = useDashboard();
  const transport = asRecord(state?.transport);
  const runtime = asRecord(state?.runtime);
  const busHealth = asRecord(transport.busHealth);
  const lastStatus = asRecord(runtime.lastStatusEvent);
  const gateway = architectureNode(state, "gateway");
  const telemetry = architectureModule(state, "telemetry");

  return (
    <ConsolePanel className="flex flex-col gap-3.5">
      <div>
        <div className="mb-2.5 flex items-center justify-between">
          <SectionHeader icon={Eye} title="Gateway / Telemetry" />
          <Chip color={toneForStatus(telemetry.status)}>{statusLabel(telemetry.status)}</Chip>
        </div>
        <div className="space-y-2 text-[11px]">
          <div className="dashboard-field dashboard-mono text-[10px] text-[var(--text-secondary)]">
            <div>control: {stringValue(busHealth.control_endpoint, "tcp://127.0.0.1:5580")}</div>
            <div className="mt-0.5">telemetry: {stringValue(busHealth.telemetry_endpoint, "tcp://127.0.0.1:5581")}</div>
          </div>
          <div className="flex justify-between items-center"><span className="dashboard-meta">Peer 연결</span><Chip color={booleanValue(transport.peerActive) ? "green" : "amber"}>{booleanValue(transport.peerActive) ? "connected" : "inactive"}</Chip></div>
          <ConsoleInfoRow label="Frame Seq" value={String(transport.frameSeq ?? "n/a")} />
          <ConsoleInfoRow label="Frame Age" value={formatMs(transport.frameAgeMs, "n/a")} />
        </div>
      </div>

      <div className="h-px w-full bg-[rgba(var(--ink-rgb),0.06)]"></div>

      <div>
        <SectionHeader icon={Layers} title="Gateway State Mirror" />
        <div className="dashboard-panel-strong p-3.5 text-[12px]">
          <ConsoleInfoRow className="mb-2" label="gateway" value={gateway.summary || "idle"} />
          <ConsoleInfoRow className="mb-2" label="telemetry" value={telemetry.summary || "inactive"} />
          <ConsoleInfoRow className="mb-2" label="last status" value={stringValue(lastStatus.state, "n/a")} />
          <div className="dashboard-field dashboard-mono mt-2 text-center text-[10px] text-[var(--text-tertiary)]">
            {stringValue(lastStatus.reason, gateway.detail || telemetry.detail || "viewer/state bridge active")}
          </div>
        </div>
      </div>
    </ConsolePanel>
  );
}

export function LogsWidget() {
  const { state } = useDashboard();
  const [logsExpanded, setLogsExpanded] = useState(false);
  const [remoteLogs, setRemoteLogs] = useState<LogRecord[]>([]);
  const [logsError, setLogsError] = useState("");
  const limit = logsExpanded ? 120 : 40;
  const fallbackLogs = recentLogs(state, limit);
  const logs = remoteLogs.length > 0 ? remoteLogs : fallbackLogs;

  useEffect(() => {
    let cancelled = false;

    async function loadLogs() {
      try {
        const response = await requestJson<{ logs: LogRecord[] }>(`/api/logs?limit=${limit}`);
        if (cancelled) {
          return;
        }
        setRemoteLogs(Array.isArray(response.logs) ? [...response.logs].reverse() : []);
        setLogsError("");
      } catch (error) {
        if (!cancelled) {
          setLogsError(error instanceof Error ? error.message : String(error));
        }
      }
    }

    void loadLogs();
    const intervalId = window.setInterval(() => {
      void loadLogs();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [limit]);

  return (
    <ConsolePanel className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <div className="dashboard-icon-shell">
            <Terminal className="size-4" />
          </div>
          <div>
            <div className="dashboard-eyebrow">system feed</div>
            <h3 className="text-[13px] font-medium text-[var(--foreground)]">시스템 로그</h3>
          </div>
        </div>
        <button
          onClick={() => setLogsExpanded((current) => !current)}
          className="dashboard-button-secondary !rounded-full !px-3 !py-2 text-[11px] text-[var(--text-secondary)]"
        >
          {logsExpanded ? "접기" : "모두 보기"}
          {logsExpanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
        </button>
      </div>

      {logsError !== "" && (
        <div className="mb-3 rounded-xl border border-[var(--tone-amber-border)] bg-[var(--tone-amber-bg)] px-3 py-2 text-[11px] text-[var(--tone-amber-fg)]">
          log refresh degraded: {logsError}
        </div>
      )}

      <div className={`dashboard-scroll dashboard-panel-strong space-y-1 dashboard-mono text-[11px] p-3 flex-1 overflow-y-auto ${logsExpanded ? "max-h-[300px]" : "max-h-[140px]"}`}>
        {logs.length === 0 && (
          <div className="text-[var(--text-faint)] px-2 py-2">no logs yet</div>
        )}
        {logs.map((log, index) => {
          const level = stringValue(log.level || log.stream, "info");
          return (
            <div key={`${log.source}-${index}`} className="flex gap-2 px-2 py-1.5 transition-colors hover:bg-[rgba(123,102,79,0.05)] rounded-xl">
              <span className="text-[var(--text-faint)] w-[70px] shrink-0">{log.timestampNs ? String(log.timestampNs).slice(-8) : log.stream}</span>
              <span className={`w-[80px] shrink-0 font-semibold ${level === "error" || level === "stderr" ? "text-red-500" : level === "warn" || level === "warning" ? "text-amber-500" : "text-[var(--text-secondary)]"}`}>
                {log.source}
              </span>
              <span className={`flex-1 truncate ${level === "error" || level === "stderr" ? "text-red-600" : level === "warn" || level === "warning" ? "text-amber-600" : "text-[var(--foreground)]"}`}>
                {log.message}
              </span>
            </div>
          );
        })}
      </div>
    </ConsolePanel>
  );
}
