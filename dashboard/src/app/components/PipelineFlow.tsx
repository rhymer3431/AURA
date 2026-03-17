import { ArrowRight } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, booleanValue, formatMs, runtimeComponentLabel, stringValue } from "../selectors";

export function PipelineFlow() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const sensors = asRecord(state?.sensors);
  const perception = asRecord(state?.perception);
  const memory = asRecord(state?.memory);
  const scratchpad = asRecord(memory.scratchpad);
  const navdp = asRecord(state?.services.navdp);
  const dual = asRecord(state?.services.dual);
  const transport = asRecord(state?.transport);
  const actionStatus = asRecord(runtime.actionStatus);
  const runtimeOwner = stringValue(runtime.ownerDisplayName, "NavigationRuntime");
  const runtimeComponent = runtimeComponentLabel(runtime.ownerComponent, "navigation_runtime");
  const plannerPhase = stringValue(runtime.interactivePhase || runtime.plannerControlMode, "idle");
  const missionState = stringValue(scratchpad.taskState, "idle");
  const missionPriority = stringValue(scratchpad.nextPriority, "idle");
  const activeCommandType = stringValue(runtime.activeCommandType, "idle") || "idle";
  const planningBackend =
    stringValue(dual.status) === "ok"
      ? "navdp + dual"
      : stringValue(navdp.status) === "ok"
        ? "navdp"
        : stringValue(navdp.status, "inactive");
  const runtimeIoDetail = booleanValue(transport.peerActive)
    ? `peer ${stringValue(transport.peerSessionId, "connected")}`
    : booleanValue(transport.viewerEnabled)
      ? "viewer publish"
      : "idle";
  const locomotionState = stringValue(actionStatus.state, activeCommandType === "idle" ? "idle" : "tracking");

  const stages = [
    {
      name: "Navigation\nRuntime",
      detail: runtimeComponent,
      ok: state?.session.active === true || stringValue(runtime.ownerComponent) !== "",
    },
    {
      name: "Observation\nModule",
      detail: formatMs(state?.transport.frameAgeMs, "idle"),
      ok: sensors.rgbAvailable === true,
    },
    {
      name: "World Model\nModule",
      detail: `${Number(memory.objectCount ?? 0)} obj / ${missionState}`,
      ok: state?.session.config?.memoryStore === true || Number(memory.objectCount ?? 0) > 0,
    },
    {
      name: "Mission\nModule",
      detail: missionPriority,
      ok: missionState !== "idle" || activeCommandType !== "idle",
    },
    {
      name: "Planning\nModule",
      detail: `${plannerPhase} / ${planningBackend}`,
      ok: stringValue(navdp.status) === "ok" && ["ok", "not_required", "inactive", ""].includes(stringValue(dual.status)),
    },
    {
      name: "Execution\nModule",
      detail: `v${Number(runtime.trajVersion ?? 0)}`,
      ok: Number(runtime.trajVersion ?? 0) > 0,
    },
    {
      name: "Runtime I/O\nModule",
      detail: runtimeIoDetail,
      ok: booleanValue(transport.viewerEnabled) || booleanValue(transport.frameAvailable) || booleanValue(transport.peerActive),
    },
    {
      name: "locomotion.\nruntime",
      detail: activeCommandType,
      ok: locomotionState !== "idle" || activeCommandType !== "idle",
    },
  ];

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-[15px] font-semibold text-black">데이터 흐름 파이프라인</h3>
          <p className="text-[12px] text-black/50 mt-0.5">
            NavigationRuntime → Observation → World Model → Mission → Planning → Execution → Runtime I/O
          </p>
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
          owner: {runtimeOwner}
        </span>
        <span className="bg-violet-50 border border-violet-200 text-violet-700 rounded-full px-3 py-1 font-medium shadow-sm">
          phase: {plannerPhase}
        </span>
        <span className="bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-full px-3 py-1 font-medium shadow-sm">
          backend: {planningBackend}
        </span>
        <span className="bg-amber-50 border border-amber-200 text-amber-700 rounded-full px-3 py-1 font-medium shadow-sm">
          detector: {stringValue(perception.detectorBackend, "off")}
        </span>
      </div>
    </div>
  );
}
