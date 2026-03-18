import { ArrowRight } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, statusTone, stringValue } from "../selectors";

export function PipelineFlow() {
  const { state } = useDashboard();
  const gateway = architectureNode(state, "gateway");
  const controlServer = architectureNode(state, "mainControlServer");
  const modules = [
    architectureModule(state, "perception"),
    architectureModule(state, "memory"),
    architectureModule(state, "s2"),
    architectureModule(state, "nav"),
    architectureModule(state, "locomotion"),
    architectureModule(state, "telemetry"),
  ];
  const core = state?.architecture.mainControlServer.core;

  const stages = [
    {
      name: "Robot\nGateway",
      detail: gateway.detail || gateway.summary,
      status: gateway.status,
    },
    {
      name: "Main Control\nServer",
      detail: controlServer.summary || controlServer.detail,
      status: controlServer.status,
    },
    ...modules.map((module) => ({
      name: module.name.includes(" ")
        ? module.name.replace(" ", "\n")
        : module.name,
      detail: module.summary || module.detail,
      status: module.status,
    })),
  ];

  const toneClass = (status: string) => {
    const tone = statusTone(status);
    if (tone === "green") {
      return "bg-emerald-500";
    }
    if (tone === "amber") {
      return "bg-amber-500";
    }
    if (tone === "red") {
      return "bg-red-500";
    }
    return "bg-slate-400";
  };

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-[15px] font-semibold text-black">Runtime Architecture Flow</h3>
          <p className="text-[12px] text-black/50 mt-0.5">robot gateway → main control server → runtime modules</p>
        </div>
      </div>

      <div className="flex items-center gap-2 overflow-x-auto pb-3 flex-1">
        {stages.map((stage, index) => (
          <div key={stage.name} className="flex items-center gap-2 shrink-0">
            <div className="bg-white rounded-2xl px-4 py-3 min-w-[110px] text-center shadow-sm transition-all">
              <span className={`inline-block size-2 rounded-full mb-1.5 ${toneClass(stage.status)}`} />
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

      <div className="mt-4">
        <div className="text-[11px] text-black/45 mb-2">Main Control Server Core</div>
        <div className="flex items-center gap-2.5 text-[11px] flex-wrap">
          {core !== undefined &&
            Object.values(core).map((node) => (
              <span
                key={node.name}
                className="bg-white border border-black/5 text-black/70 rounded-full px-3 py-1 font-medium shadow-sm"
              >
                {node.name}: {stringValue(node.summary, "idle")}
              </span>
            ))}
        </div>
      </div>
    </div>
  );
}
