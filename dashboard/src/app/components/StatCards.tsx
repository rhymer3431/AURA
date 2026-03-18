import { TrendingUp, TrendingDown } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, asRecord, formatMeters, formatMs, statusLabel, stringValue } from "../selectors";

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
      bg: "bg-[#E3E5FE]",
    },
    {
      label: "Control Server",
      value: controlServer.summary || "Ready",
      change: stringValue(controlServer.detail, "idle"),
      trend: controlServer.status === "ok" ? ("up" as const) : ("down" as const),
      bg: "bg-[#E3F1FC]",
    },
    {
      label: "Modules Ready",
      value: `${healthyModules} / ${requiredModules.length}`,
      change: `${modules.filter((item) => item.status === "ok").length} active`,
      trend: healthyModules === requiredModules.length && requiredModules.length > 0 ? ("up" as const) : ("down" as const),
      bg: "bg-[#EFE8FC]",
    },
    {
      label: "Recovery State",
      value: recoveryState,
      change: formatMeters(runtime.goalDistanceM, stringValue(runtime.activeCommandType, "idle")),
      trend: recoveryState === "NORMAL" ? ("up" as const) : ("down" as const),
      bg: "bg-[#E6F2FA]",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
      {stats.map((item) => (
        <div
          key={item.label}
          className={`${item.bg} rounded-3xl p-6 flex flex-col justify-between h-[120px] transition-transform hover:scale-[1.02]`}
        >
          <div className="text-[14px] font-medium text-black/80">{item.label}</div>
          <div className="flex items-end justify-between gap-3">
            <div className="text-[24px] font-semibold text-black leading-none tracking-tight break-all">
              {item.value}
            </div>
            <div className="flex items-center gap-1 text-[12px] font-medium text-black/70 mb-1">
              {item.change}
              {item.trend === "up" ? (
                <TrendingUp className="size-3 text-black/80" />
              ) : (
                <TrendingDown className="size-3 text-black/80" />
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
