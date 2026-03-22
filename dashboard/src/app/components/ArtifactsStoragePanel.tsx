import { Database, FolderOpen, HardDrive, Radio } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, stringValue } from "../selectors";
import { ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[12px]">
      <span className="dashboard-meta">{label}</span>
      <span className="text-[var(--foreground)] font-medium text-right break-all">{value}</span>
    </div>
  );
}

export function ArtifactsStoragePanel() {
  const { bootstrap, state } = useDashboard();
  const memory = asRecord(state?.memory);
  const scratchpad = asRecord(memory.scratchpad);
  const transport = asRecord(state?.transport);
  const busHealth = asRecord(transport.busHealth);
  const logFiles = (state?.processes ?? []).flatMap((process) => [
    { key: `${process.name}-stdout`, label: `${process.name} stdout`, path: process.stdoutLog },
    { key: `${process.name}-stderr`, label: `${process.name} stderr`, path: process.stderrLog },
  ]);
  const logSources = Array.from(new Set((state?.logs ?? []).map((item) => item.source).filter((item) => item !== ""))).slice(-6);

  return (
    <ConsolePanel className="flex flex-col gap-5">
      <ConsoleSectionTitle
        icon={HardDrive}
        eyebrow="artifact mirror"
        title="Artifacts & Diagnostics"
        description="runtime artifact, memory footprint, transport endpoint, and implementation log path를 한 곳에 모았습니다."
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="dashboard-panel-strong p-5">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--text-secondary)] mb-3">
            <Radio className="size-4 text-[var(--text-faint)]" />
            Runtime Endpoints
          </div>
          <div className="space-y-3">
            <DetailRow label="API Base" value={bootstrap?.apiBaseUrl ?? "n/a"} />
            <DetailRow label="WebRTC Base" value={bootstrap?.webrtcBasePath ?? "n/a"} />
            <DetailRow label="Control" value={stringValue(busHealth.control_endpoint, "tcp://127.0.0.1:5580")} />
            <DetailRow label="Telemetry" value={stringValue(busHealth.telemetry_endpoint, "tcp://127.0.0.1:5581")} />
          </div>
        </div>

        <div className="dashboard-panel-strong p-5">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--text-secondary)] mb-3">
            <Database className="size-4 text-[var(--text-faint)]" />
            Memory Module Footprint
          </div>
          <div className="grid grid-cols-3 gap-3 mb-4">
            {[
              { label: "Objects", value: String(memory.objectCount ?? 0) },
              { label: "Places", value: String(memory.placeCount ?? 0) },
              { label: "Rules", value: String(memory.semanticRuleCount ?? 0) },
            ].map((item) => (
              <div key={item.label} className="dashboard-field text-center">
                <div className="dashboard-eyebrow mb-1">{item.label}</div>
                <div className="dashboard-mono text-[16px] font-semibold text-[var(--foreground)]">{item.value}</div>
              </div>
            ))}
          </div>
          <DetailRow label="Scratchpad State" value={stringValue(scratchpad.taskState, "idle")} />
          <div className="dashboard-field dashboard-mono mt-3 text-[12px] text-[var(--text-secondary)] leading-relaxed break-words">
            {stringValue(scratchpad.instruction, "no active scratchpad instruction")}
          </div>
        </div>

        <div className="dashboard-panel-strong p-5">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--text-secondary)] mb-3">
            <FolderOpen className="size-4 text-[var(--text-faint)]" />
            Runtime Mirror Snapshot
          </div>
          <div className="space-y-3">
            <DetailRow label="Scene Preset" value={state?.session.config?.scenePreset ?? "inactive"} />
            <DetailRow label="Peer Session" value={stringValue(transport.peerSessionId, "none")} />
            <DetailRow label="Last Event" value={state?.session.lastEvent?.message ?? "n/a"} />
            <DetailRow label="Recent Sources" value={logSources.length > 0 ? logSources.join(", ") : "n/a"} />
          </div>
        </div>
      </div>

      <div className="dashboard-panel-strong p-5">
        <div className="dashboard-title text-[13px] mb-3">Implementation Log Files</div>
        <div className="dashboard-scroll space-y-2 max-h-[320px] overflow-y-auto pr-1">
          {logFiles.length === 0 && (
            <div className="dashboard-field text-[12px] text-[var(--text-tertiary)]">no process log files available</div>
          )}
          {logFiles.map((item) => (
            <div key={item.key} className="dashboard-field">
              <div className="text-[11px] font-medium text-[var(--text-secondary)] mb-1">{item.label}</div>
              <div className="dashboard-mono text-[11px] text-[var(--text-tertiary)] break-all">{item.path || "n/a"}</div>
            </div>
          ))}
        </div>
      </div>
    </ConsolePanel>
  );
}
