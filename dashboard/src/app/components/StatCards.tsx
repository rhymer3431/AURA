import { TrendingUp, TrendingDown } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, asRecord, formatMeters, formatMs, statusLabel, stringValue } from "../selectors";
import { ConsoleMetricCard, type ConsoleTone } from "./console-ui";

export function StatCards() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
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
  const requiredModules = modules.filter((item) => item.required);
  const healthyModules = requiredModules.filter((item) => item.status === "ok").length;
  const recoveryState = stringValue(runtime.recoveryState, "NORMAL");

  const stats = [
    {
      label: "Gateway",
      value: statusLabel(gateway.status),
      change: gateway.detail || "idle",
      trend: gateway.status === "ok" ? ("up" as const) : ("down" as const),
      tone: (gateway.status === "ok" ? "cyan" : "amber") as ConsoleTone,
    },
    {
      label: "Control Server",
      value: controlServer.summary || "Ready",
      change: stringValue(controlServer.detail, "idle"),
      trend: controlServer.status === "ok" ? ("up" as const) : ("down" as const),
      tone: (controlServer.status === "ok" ? "emerald" : "amber") as ConsoleTone,
    },
    {
      label: "Modules Ready",
      value: `${healthyModules} / ${requiredModules.length}`,
      change: `${modules.filter((item) => item.status === "ok").length} active`,
      trend: healthyModules === requiredModules.length && requiredModules.length > 0 ? ("up" as const) : ("down" as const),
      tone: (healthyModules === requiredModules.length && requiredModules.length > 0 ? "violet" : "amber") as ConsoleTone,
    },
    {
      label: "Recovery State",
      value: recoveryState,
      change: formatMeters(runtime.goalDistanceM, stringValue(runtime.activeCommandType, "idle")),
      trend: recoveryState === "NORMAL" ? ("up" as const) : ("down" as const),
      tone: (recoveryState === "NORMAL" ? "emerald" : "coral") as ConsoleTone,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 2xl:grid-cols-4">
      {stats.map((item) => (
        <ConsoleMetricCard
          key={item.label}
          label={item.label}
          value={item.value}
          tone={item.tone}
          className="transition-colors duration-150 hover:bg-[rgba(255,255,255,0.98)]"
          valueClassName="break-all"
          meta={(
            <div className="flex items-center justify-between gap-3 border-t border-[rgba(17,23,28,0.06)] pt-3">
              <span className="truncate">{item.change}</span>
              {item.trend === "up" ? (
                <TrendingUp className="size-3 shrink-0 text-[var(--text-secondary)]" />
              ) : (
                <TrendingDown className="size-3 shrink-0 text-[var(--text-secondary)]" />
              )}
            </div>
          )}
        />
      ))}
    </div>
  );
}
