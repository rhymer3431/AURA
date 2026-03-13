import { Database, FolderOpen, HardDrive, Radio } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, stringValue } from "../selectors";

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[12px]">
      <span className="text-black/45">{label}</span>
      <span className="text-black/75 font-medium text-right break-all">{value}</span>
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
    <div className="bg-[#F7F9FB] rounded-3xl p-6 flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <HardDrive className="size-4 text-black/40" />
        <div>
          <h3 className="text-[15px] font-semibold text-black">Artifacts & Storage</h3>
          <p className="text-[12px] text-black/50 mt-0.5">운영 중 생성되는 로그 경로, memory footprint, transport endpoint를 따로 모았습니다.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-black/75 mb-3">
            <Radio className="size-4 text-black/35" />
            Connected Endpoints
          </div>
          <div className="space-y-3">
            <DetailRow label="API Base" value={bootstrap?.apiBaseUrl ?? "n/a"} />
            <DetailRow label="WebRTC Base" value={bootstrap?.webrtcBasePath ?? "n/a"} />
            <DetailRow label="Control" value={stringValue(busHealth.control_endpoint, "tcp://127.0.0.1:5580")} />
            <DetailRow label="Telemetry" value={stringValue(busHealth.telemetry_endpoint, "tcp://127.0.0.1:5581")} />
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-black/75 mb-3">
            <Database className="size-4 text-black/35" />
            Memory Footprint
          </div>
          <div className="grid grid-cols-3 gap-3 mb-4">
            {[
              { label: "Objects", value: String(memory.objectCount ?? 0) },
              { label: "Places", value: String(memory.placeCount ?? 0) },
              { label: "Rules", value: String(memory.semanticRuleCount ?? 0) },
            ].map((item) => (
              <div key={item.label} className="rounded-xl bg-[#F7F9FB] px-3 py-3 text-center">
                <div className="text-[11px] text-black/45 mb-1">{item.label}</div>
                <div className="text-[16px] font-semibold text-black">{item.value}</div>
              </div>
            ))}
          </div>
          <DetailRow label="Scratchpad State" value={stringValue(scratchpad.taskState, "idle")} />
          <div className="mt-3 rounded-xl bg-[#F7F9FB] px-3 py-3 text-[12px] text-black/65 leading-relaxed break-words">
            {stringValue(scratchpad.instruction, "no active scratchpad instruction")}
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-semibold text-black/75 mb-3">
            <FolderOpen className="size-4 text-black/35" />
            Runtime Snapshot
          </div>
          <div className="space-y-3">
            <DetailRow label="Scene Preset" value={state?.session.config?.scenePreset ?? "inactive"} />
            <DetailRow label="Peer Session" value={stringValue(transport.peerSessionId, "none")} />
            <DetailRow label="Last Event" value={state?.session.lastEvent?.message ?? "n/a"} />
            <DetailRow label="Recent Sources" value={logSources.length > 0 ? logSources.join(", ") : "n/a"} />
          </div>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-5 shadow-sm">
        <div className="text-[12px] font-semibold text-black/75 mb-3">Process Log Files</div>
        <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
          {logFiles.length === 0 && (
            <div className="rounded-xl bg-[#F7F9FB] px-3 py-3 text-[12px] text-black/45">no process log files available</div>
          )}
          {logFiles.map((item) => (
            <div key={item.key} className="rounded-xl bg-[#F7F9FB] px-3 py-3">
              <div className="text-[11px] font-medium text-black/65 mb-1">{item.label}</div>
              <div className="font-mono text-[11px] text-black/55 break-all">{item.path || "n/a"}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
