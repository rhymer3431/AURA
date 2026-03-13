import { ArrowRight } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, booleanValue, formatMs, stringValue } from "../selectors";

export function PipelineFlow() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const sensors = asRecord(state?.sensors);
  const perception = asRecord(state?.perception);
  const memory = asRecord(state?.memory);
  const navdp = asRecord(state?.services.navdp);
  const dual = asRecord(state?.services.dual);

  const stages = [
    {
      name: "Sensor\nCapture",
      detail: formatMs(state?.transport.frameAgeMs, "idle"),
      ok: sensors.rgbAvailable === true,
    },
    {
      name: "Supervisor\nprocess_frame",
      detail: stringValue(sensors.source, "idle"),
      ok: state?.session.active === true,
    },
    {
      name: "Perception\nPipeline",
      detail: stringValue(perception.detectorBackend, "off"),
      ok: perception.detectorReady === true || Number(perception.detectionCount ?? 0) >= 0,
    },
    {
      name: "Memory\nWrite",
      detail: booleanValue(memory.memoryAwareTaskActive) ? "task_active" : "idle",
      ok: state?.session.config?.memoryStore === true,
    },
    {
      name: "NavDP\nService",
      detail: formatMs(navdp.latencyMs, stringValue(navdp.status, "down")),
      ok: stringValue(navdp.status) === "ok",
    },
    {
      name: "Dual / S2\nPath",
      detail: stringValue(dual.status, "inactive"),
      ok: ["ok", "not_required", "inactive"].includes(stringValue(dual.status)),
    },
    {
      name: "Trajectory\nUpdate",
      detail: `v${Number(runtime.trajVersion ?? 0)}`,
      ok: Number(runtime.trajVersion ?? 0) > 0,
    },
    {
      name: "Command\nVector",
      detail: stringValue(runtime.activeCommandType, "idle"),
      ok: stringValue(runtime.activeCommandType) !== "",
    },
  ];

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-[15px] font-semibold text-black">데이터 흐름 파이프라인</h3>
          <p className="text-[12px] text-black/50 mt-0.5">sensor capture → planner → command vector</p>
        </div>
      </div>

      <div className="flex items-center gap-2 overflow-x-auto pb-3 flex-1">
        {stages.map((stage, index) => (
          <div key={stage.name} className="flex items-center gap-2 shrink-0">
            <div className="bg-white rounded-2xl px-4 py-3 min-w-[110px] text-center shadow-sm transition-all">
              <span
                className={`inline-block size-2 rounded-full mb-1.5 ${stage.ok ? "bg-emerald-500" : "bg-amber-500"}`}
              />
              <div className="text-[12px] font-medium text-black/80 whitespace-pre-line leading-tight">
                {stage.name}
              </div>
              <div className="text-[10px] text-black/45 mt-1">{stage.detail}</div>
            </div>
            {index < stages.length - 1 && (
              <ArrowRight className="size-4 text-black/20 shrink-0 mx-1" />
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-2.5 text-[11px] flex-wrap">
        <span className="bg-sky-50 border border-sky-200 text-sky-700 rounded-full px-3 py-1 font-medium shadow-sm">
          phase: {stringValue(runtime.interactivePhase || runtime.plannerControlMode, "idle")}
        </span>
        <span className="bg-violet-50 border border-violet-200 text-violet-700 rounded-full px-3 py-1 font-medium shadow-sm">
          cmd: {stringValue(runtime.activeCommandType, "none")}
        </span>
        <span className="bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-full px-3 py-1 font-medium shadow-sm">
          traj v{Number(runtime.trajVersion ?? 0)}
        </span>
      </div>
    </div>
  );
}
