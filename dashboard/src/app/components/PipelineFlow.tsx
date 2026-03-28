import { ArrowRight } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, statusTone, stringValue } from "../selectors";
import { ConsolePanel, toneFromStatusTone } from "./console-ui";

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
    <ConsolePanel className="h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-[15px] font-semibold text-[var(--foreground)]">데이터 흐름 파이프라인</h3>
        <p className="mt-1 text-[12px] text-[var(--text-tertiary)]">sensor capture → command vector</p>
      </div>

      <div className="flex flex-1 items-center gap-2 overflow-x-auto pb-3">
        {stages.map((stage, index) => (
          <div key={stage.name} className="flex shrink-0 items-center gap-2">
            <div className="dashboard-flow-card min-w-[104px] px-4 py-3 text-center">
              <span className={`mb-1.5 inline-block size-2 rounded-full ${toneClass(stage.status)}`} />
              <div className="text-[12px] font-medium whitespace-pre-line leading-tight text-[var(--foreground)]">
                {stage.name}
              </div>
              <div className="dashboard-micro mt-1.5">{stage.detail}</div>
            </div>
            {index < stages.length - 1 && (
              <ArrowRight className="mx-0.5 size-4 shrink-0 text-[var(--text-faint)]" />
            )}
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2.5 text-[11px]">
        <div className="dashboard-flow-note dashboard-flow-note--cyan">gateway → main control server</div>
        <div className="dashboard-flow-note dashboard-flow-note--violet">runtime module mirror</div>
        <div className="flex-1" />
        <div className="flex items-center gap-2.5 flex-wrap">
          {core !== undefined &&
            Object.values(core).map((node) => (
              <span
                key={node.name}
                className={`dashboard-flow-core-chip dashboard-flow-core-chip--${toneFromStatusTone(statusTone(node.status))}`}
              >
                {node.name}: {stringValue(node.summary, "idle")}
              </span>
            ))}
        </div>
      </div>
    </ConsolePanel>
  );
}
