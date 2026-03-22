import { ArrowRight } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, statusTone, stringValue } from "../selectors";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle, toneFromStatusTone } from "./console-ui";

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
      <ConsoleSectionTitle
        icon={ArrowRight}
        eyebrow="signal path"
        title="Runtime Architecture Flow"
        description="robot gateway → main control server → runtime modules"
      />

      <div className="flex items-center gap-2 overflow-x-auto pb-3 flex-1">
        {stages.map((stage, index) => (
          <div key={stage.name} className="flex items-center gap-2 shrink-0">
            <div className="dashboard-panel-strong min-w-[118px] px-4 py-4 text-center transition-all">
              <span className={`inline-block size-2 rounded-full mb-2 ${toneClass(stage.status)}`} />
              <div className="text-[12px] font-medium whitespace-pre-line leading-tight text-[var(--foreground)]">
                {stage.name}
              </div>
              <div className="dashboard-micro mt-2">{stage.detail}</div>
            </div>
            {index < stages.length - 1 && (
              <ArrowRight className="size-4 text-[var(--text-faint)] shrink-0 mx-1" />
            )}
          </div>
        ))}
      </div>

      <div className="mt-4">
        <div className="dashboard-eyebrow mb-2">Main Control Server Core</div>
        <div className="flex items-center gap-2.5 text-[11px] flex-wrap">
          {core !== undefined &&
            Object.values(core).map((node) => (
              <ConsoleBadge
                key={node.name}
                tone={toneFromStatusTone(statusTone(node.status))}
                className="!rounded-[14px]"
              >
                {node.name}: {stringValue(node.summary, "idle")}
              </ConsoleBadge>
            ))}
        </div>
      </div>
    </ConsolePanel>
  );
}
