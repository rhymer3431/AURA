import { TrendingUp, TrendingDown } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, formatMeters, formatMs, stringValue } from "../selectors";

export function StatCards() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const perception = asRecord(state?.perception);
  const navdp = asRecord(state?.services.navdp);
  const activeProcesses = state?.processes.filter((item) => item.state === "running").length ?? 0;
  const requiredProcesses = state?.processes.filter((item) => item.required).length ?? 0;
  const plannerPhase = stringValue(runtime.interactivePhase || runtime.plannerControlMode, "idle");
  const detectionCount = Number(perception.detectionCount ?? 0);

  const stats = [
    {
      label: "Active Processes",
      value: `${activeProcesses} / ${requiredProcesses}`,
      change: state?.session.active ? "running" : "idle",
      trend: "up" as const,
      bg: "bg-[#E3E5FE]",
    },
    {
      label: "Planner Phase",
      value: plannerPhase,
      change: formatMeters(runtime.goalDistanceM, "n/a"),
      trend: Number(runtime.staleSec ?? 0) > 1.0 ? ("down" as const) : ("up" as const),
      bg: "bg-[#E3F1FC]",
    },
    {
      label: "Perception",
      value: `${detectionCount} det.`,
      change: stringValue(perception.detectorBackend, "unknown"),
      trend: detectionCount > 0 ? ("up" as const) : ("down" as const),
      bg: "bg-[#EFE8FC]",
    },
    {
      label: "Ext. Services",
      value: formatMs(navdp.latencyMs, "n/a"),
      change: stringValue(navdp.status, "unknown"),
      trend: stringValue(navdp.status) === "ok" ? ("up" as const) : ("down" as const),
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
