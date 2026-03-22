import { TrendingUp, TrendingDown } from "lucide-react";

import { useDashboard } from "../state";
import { architectureNode, asRecord, coreNode, formatMs, numberValue, statusLabel, stringValue } from "../selectors";
import { ConsoleMetricCard, type ConsoleTone } from "./console-ui";

export function StatCards() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const plannerCoordinator = coreNode(state, "plannerCoordinator");
  const perception = asRecord(state?.perception);
  const serviceSnapshots = [
    asRecord(state?.services.navdp),
    asRecord(state?.services.dual),
    asRecord(state?.services.system2),
  ];
  const runningProcesses = (state?.processes ?? []).filter((item) =>
    ["running", "ok", "connected", "starting"].includes(item.state),
  ).length;
  const processTotal = (state?.processes ?? []).length;
  const plannerLatencyMs =
    plannerCoordinator.latencyMs
    ?? numberValue(asRecord(plannerCoordinator.metrics).latencyMs)
    ?? numberValue(runtime.plannerLatencyMs)
    ?? numberValue(runtime.planLatencyMs)
    ?? null;
  const detectionCount = numberValue(perception.detectionCount) ?? 0;
  const trackedCount = numberValue(perception.trackedDetectionCount) ?? 0;
  const externalLatencyMs =
    serviceSnapshots
      .map((item) => numberValue(item.latencyMs))
      .find((value) => value !== null)
    ?? architectureNode(state, "gateway").latencyMs
    ?? numberValue(asRecord(state?.transport).frameAgeMs)
    ?? null;

  const stats = [
    {
      label: "Active Processes",
      value: `${runningProcesses} / ${processTotal}`,
      change: `${(state?.processes ?? []).filter((item) => item.required).length} required`,
      trend: runningProcesses > 0 ? ("up" as const) : ("down" as const),
      tone: "violet" as ConsoleTone,
    },
    {
      label: "Planner Phase",
      value: plannerLatencyMs === null ? stringValue(plannerCoordinator.summary, "idle") : formatMs(plannerLatencyMs, "n/a"),
      change: stringValue(plannerCoordinator.detail, plannerCoordinator.summary || "planner idle"),
      trend: plannerCoordinator.status === "ok" ? ("up" as const) : ("down" as const),
      tone: "cyan" as ConsoleTone,
    },
    {
      label: "Perception",
      value: `${detectionCount} det.`,
      change: `${trackedCount} tracked`,
      trend: detectionCount > 0 ? ("up" as const) : ("down" as const),
      tone: "violet" as ConsoleTone,
    },
    {
      label: "Ext. Services",
      value: externalLatencyMs === null ? statusLabel(stringValue(serviceSnapshots[0].status, "inactive")) : formatMs(externalLatencyMs, "n/a"),
      change: serviceSnapshots.map((item) => stringValue(item.name)).filter((item) => item !== "").join(" · ") || "gateway mirror",
      trend: externalLatencyMs !== null || serviceSnapshots.some((item) => stringValue(item.status) === "ok") ? ("up" as const) : ("down" as const),
      tone: "cyan" as ConsoleTone,
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
          className="transition-transform duration-150 hover:-translate-y-0.5"
          valueClassName="break-all"
          meta={(
            <div className="flex items-center justify-between gap-3 border-t border-[rgba(var(--ink-rgb),0.07)] pt-3">
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
