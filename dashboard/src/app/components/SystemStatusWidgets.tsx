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
  asRecord,
  booleanValue,
  formatMs,
  recentLogs,
  runtimeComponentLabel,
  statusLabel,
  statusTone,
  stringValue,
} from "../selectors";

function SectionHeader({ icon: Icon, title }: { icon: typeof Scan; title: string }) {
  return (
    <div className="flex items-center gap-1.5 mb-3">
      <Icon className="size-4 text-black/40" />
      <span className="text-[13px] font-medium text-black">{title}</span>
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
    green: "bg-emerald-50 text-emerald-600 border-emerald-200",
    blue: "bg-sky-50 text-sky-600 border-sky-200",
    amber: "bg-amber-50 text-amber-600 border-amber-200",
    violet: "bg-violet-50 text-violet-600 border-violet-200",
    slate: "bg-slate-50 text-slate-600 border-slate-200",
  };
  const dots = {
    green: "bg-emerald-500",
    blue: "bg-sky-500",
    amber: "bg-amber-500",
    violet: "bg-violet-500",
    slate: "bg-slate-500",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] border ${tones[color]}`}>
      <span className={`size-1.5 rounded-full ${dots[color]}`} />
      {children}
    </span>
  );
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
    <div className="bg-[#F7F9FB] rounded-3xl p-6">
      <SectionHeader icon={Box} title="프로세스 구성" />
      <div className="space-y-2.5">
        {processes.map((process) => (
          <div
            key={process.name}
            className="flex items-center gap-3 bg-white rounded-2xl px-4 py-3 shadow-sm transition-all"
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
              <div className="text-[12px] text-black/80 truncate">{process.name}</div>
              <div className="text-[10px] text-black/40">
                {statusLabel(process.state)} · PID {process.pid ?? "n/a"}
              </div>
            </div>
            <Chip color={toneForStatus(process.state)}>{statusLabel(process.state)}</Chip>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SensorsWidget() {
  const { state } = useDashboard();
  const sensors = asRecord(state?.sensors);
  const frameSource = runtimeComponentLabel(
    sensors.source,
    runtimeComponentLabel(state?.runtime?.ownerComponent, "navigation_runtime"),
  );

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6">
      <SectionHeader icon={MonitorSmartphone} title="Observation 입력" />
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: "RGB", ok: booleanValue(sensors.rgbAvailable) },
          { label: "Depth", ok: booleanValue(sensors.depthAvailable) },
          { label: "Pose", ok: booleanValue(sensors.poseAvailable) },
        ].map((sensor) => (
          <div key={sensor.label} className="bg-white rounded-2xl py-3 text-center shadow-sm">
            <span className={`inline-block size-2.5 rounded-full ${sensor.ok ? "bg-emerald-500" : "bg-amber-500"} mb-1.5`} />
            <div className="text-[12px] font-medium text-black/70">{sensor.label}</div>
          </div>
        ))}
      </div>
      <div className="space-y-1.5 text-[11px]">
        <div className="flex justify-between items-center"><span className="text-black/50">Frame Freshness</span><span className="text-emerald-600 font-medium">{formatMs(state?.transport.frameAgeMs, "n/a")}</span></div>
        <div className="flex justify-between items-center"><span className="text-black/50">Frame Source</span><span className="text-black/70 truncate ml-1 bg-black/[0.03] px-1.5 py-0.5 rounded">{frameSource}</span></div>
        <div className="flex justify-between items-center"><span className="text-black/50">Frame ID</span><span className="text-black/70">{String(sensors.frameId ?? "n/a")}</span></div>
      </div>
    </div>
  );
}

export function PerceptionWidget() {
  const { state } = useDashboard();
  const perception = asRecord(state?.perception);
  const capability = asRecord(perception.detectorCapability);

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6">
      <SectionHeader icon={Scan} title="Observation / Inference" />
      <div className="flex items-center justify-between text-[11px] mb-3">
        <span className="text-black/50">Detector Backend</span>
        <Chip color={booleanValue(perception.detectorReady) ? "green" : "amber"}>
          {stringValue(perception.detectorBackend, stringValue(capability.backend_name, "unknown"))}
        </Chip>
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {[
          { label: "Detection", value: String(perception.detectionCount ?? 0) },
          { label: "Tracked", value: String(perception.trackedDetectionCount ?? 0) },
          { label: "Trajectory", value: String(perception.trajectoryPointCount ?? 0) },
          { label: "Ready", value: booleanValue(perception.detectorReady) ? "yes" : "no" },
        ].map((card) => (
          <div key={card.label} className="bg-white rounded-2xl px-3 py-2.5 shadow-sm">
            <div className="text-[11px] text-black/50 mb-0.5">{card.label}</div>
            <div className="text-black font-semibold text-[16px]">{card.value}</div>
          </div>
        ))}
      </div>
      <div className="space-y-1.5 text-[11px]">
        <div className="flex justify-between items-center"><span className="text-black/50">Selected Reason</span><span className="text-black/70 bg-black/[0.03] px-1.5 py-0.5 rounded">{stringValue(perception.detectorSelectedReason, "n/a")}</span></div>
        <div className="flex justify-between items-center"><span className="text-black/50">Capability Status</span><Chip color={stringValue(capability.status) === "ready" ? "green" : "amber"}>{stringValue(capability.status, "unknown")}</Chip></div>
      </div>
    </div>
  );
}

export function MemoryWidget() {
  const { state } = useDashboard();
  const memory = asRecord(state?.memory);
  const scratchpad = asRecord(memory.scratchpad);

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <SectionHeader icon={Database} title="World Model / Memory" />
      <div className="space-y-2 text-[11px] mb-3">
        <div className="flex justify-between items-center"><span className="text-black/50">Memory-Aware</span><Chip color={booleanValue(memory.memoryAwareTaskActive) ? "blue" : "slate"}>{booleanValue(memory.memoryAwareTaskActive) ? "task_active" : "idle"}</Chip></div>
        <div className="flex justify-between items-center"><span className="text-black/50">Scratchpad</span><Chip color={stringValue(scratchpad.taskState) === "active" ? "green" : "amber"}>{stringValue(scratchpad.taskState, "idle")}</Chip></div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: "Objects", value: String(memory.objectCount ?? 0) },
          { label: "Places", value: String(memory.placeCount ?? 0) },
          { label: "Rules", value: String(memory.semanticRuleCount ?? 0) },
        ].map((card) => (
          <div key={card.label} className="bg-white rounded-2xl px-3 py-2.5 text-center shadow-sm">
            <div className="text-[11px] text-black/50 mb-0.5">{card.label}</div>
            <div className="text-black font-semibold text-[16px]">{card.value}</div>
          </div>
        ))}
      </div>
      <div className="bg-white rounded-2xl p-3 flex-1 shadow-sm">
        <div className="text-[11px] font-medium text-black/60 mb-2">Scratchpad (Current Task)</div>
        <div className="text-black/70 text-[12px] font-mono bg-[#F7F9FB] rounded-xl p-2.5 leading-tight break-words">
          {stringValue(scratchpad.instruction, "idle")}
        </div>
      </div>
      <div className="bg-sky-50 border border-sky-100 rounded-lg px-2 py-1.5 mt-2 text-[10px] text-sky-700 text-center">
        next priority: {stringValue(scratchpad.nextPriority, "n/a")}
      </div>
    </div>
  );
}

export function IpcOrchestrationWidget() {
  const { state } = useDashboard();
  const transport = asRecord(state?.transport);
  const runtime = asRecord(state?.runtime);
  const busHealth = asRecord(transport.busHealth);
  const lastStatus = asRecord(runtime.lastStatusEvent);
  const ownerDisplayName = stringValue(runtime.ownerDisplayName, "NavigationRuntime");
  const ownerComponent = runtimeComponentLabel(runtime.ownerComponent, "navigation_runtime");

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 flex flex-col gap-4">
      <div>
        <SectionHeader icon={Eye} title="IPC / Viewer Transport" />
        <div className="space-y-2 text-[11px]">
          <div className="bg-black/[0.02] border border-black/[0.04] rounded-lg px-2.5 py-2 font-mono text-[10px] text-black/60">
            <div>control: {stringValue(busHealth.control_endpoint, "tcp://127.0.0.1:5580")}</div>
            <div className="mt-0.5">telemetry: {stringValue(busHealth.telemetry_endpoint, "tcp://127.0.0.1:5581")}</div>
          </div>
          <div className="flex justify-between items-center"><span className="text-black/50">Peer 연결</span><Chip color={booleanValue(transport.peerActive) ? "green" : "amber"}>{booleanValue(transport.peerActive) ? "connected" : "inactive"}</Chip></div>
          <div className="flex justify-between items-center"><span className="text-black/50">Frame Seq</span><span className="text-black/70">{String(transport.frameSeq ?? "n/a")}</span></div>
          <div className="flex justify-between items-center"><span className="text-black/50">Frame Age</span><span className="text-black/70">{formatMs(transport.frameAgeMs, "n/a")}</span></div>
        </div>
      </div>

      <div className="h-px bg-black/5 w-full"></div>

      <div>
        <SectionHeader icon={Layers} title="Orchestration" />
        <div className="bg-white rounded-2xl p-4 shadow-sm text-[12px]">
          <div className="flex justify-between mb-2">
            <span className="text-black/50">runtime_owner</span>
            <span className="text-black/80 font-medium">{ownerDisplayName}</span>
          </div>
          <div className="flex justify-between mb-2">
            <span className="text-black/50">owner_component</span>
            <span className="text-black/80 font-medium">{ownerComponent}</span>
          </div>
          <div className="flex justify-between mb-2">
            <span className="text-black/50">planning_phase</span>
            <span className="text-black/80 font-medium">{stringValue(runtime.plannerControlMode, "idle")}</span>
          </div>
          <div className="flex justify-between mb-2">
            <span className="text-black/50">last status</span>
            <span className="text-black/70">{stringValue(lastStatus.state, "n/a")}</span>
          </div>
          <div className="text-[10px] text-black/30 mt-2 bg-white px-1.5 py-1 rounded border border-black/5 text-center">
            {stringValue(lastStatus.reason, "viewer/state bridge active")}
          </div>
        </div>
      </div>
    </div>
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
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <Terminal className="size-4 text-black/40" />
          <h3 className="text-[13px] font-medium text-black">시스템 로그</h3>
        </div>
        <button
          onClick={() => setLogsExpanded((current) => !current)}
          className="text-[11px] text-black/40 hover:text-black/80 transition-colors flex items-center gap-1 bg-black/[0.03] px-2 py-1 rounded-md"
        >
          {logsExpanded ? "접기" : "모두 보기"}
          {logsExpanded ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
        </button>
      </div>

      {logsError !== "" && (
        <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-700">
          log refresh degraded: {logsError}
        </div>
      )}

      <div className={`space-y-1 font-mono text-[11px] bg-white rounded-2xl p-3 flex-1 overflow-y-auto shadow-sm ${logsExpanded ? "max-h-[300px]" : "max-h-[140px]"}`}>
        {logs.length === 0 && (
          <div className="text-black/35 px-2 py-2">no logs yet</div>
        )}
        {logs.map((log, index) => {
          const level = stringValue(log.level || log.stream, "info");
          const sourceLabel = runtimeComponentLabel(log.source, stringValue(log.source, "runtime"));
          return (
            <div key={`${log.source}-${index}`} className="flex gap-2 py-1.5 hover:bg-[#F7F9FB] px-2 rounded-xl transition-colors">
              <span className="text-black/30 w-[70px] shrink-0">{log.timestampNs ? String(log.timestampNs).slice(-8) : log.stream}</span>
              <span className={`w-[80px] shrink-0 font-semibold ${level === "error" || level === "stderr" ? "text-red-500" : level === "warn" || level === "warning" ? "text-amber-500" : "text-black/50"}`}>
                {sourceLabel}
              </span>
              <span className={`flex-1 truncate ${level === "error" || level === "stderr" ? "text-red-600" : level === "warn" || level === "warning" ? "text-amber-600" : "text-black/70"}`}>
                {log.message}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
