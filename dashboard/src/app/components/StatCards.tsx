import { motion } from "motion/react";
import { TrendingUp, TrendingDown } from "lucide-react";

import { useDashboard } from "../state";
import { architectureNode, asRecord, coreNode, formatMs, numberValue, statusLabel, stringValue } from "../selectors";

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
      bg: "var(--tone-violet-bg)",
    },
    {
      label: "Planner Phase",
      value: plannerLatencyMs === null ? stringValue(plannerCoordinator.summary, "idle") : formatMs(plannerLatencyMs, "n/a"),
      change: stringValue(plannerCoordinator.detail, plannerCoordinator.summary || "planner idle"),
      trend: plannerCoordinator.status === "ok" ? ("up" as const) : ("down" as const),
      bg: "var(--tone-cyan-bg)",
    },
    {
      label: "Perception",
      value: `${detectionCount} det.`,
      change: `${trackedCount} tracked`,
      trend: detectionCount > 0 ? ("up" as const) : ("down" as const),
      bg: "color-mix(in srgb, var(--tone-violet-bg) 78%, white)",
    },
    {
      label: "Ext. Services",
      value: externalLatencyMs === null ? statusLabel(stringValue(serviceSnapshots[0].status, "inactive")) : formatMs(externalLatencyMs, "n/a"),
      change: serviceSnapshots.map((item) => stringValue(item.name)).filter((item) => item !== "").join(" · ") || "gateway mirror",
      trend: externalLatencyMs !== null || serviceSnapshots.some((item) => stringValue(item.status) === "ok") ? ("up" as const) : ("down" as const),
      bg: "color-mix(in srgb, var(--tone-cyan-bg) 78%, white)",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {stats.map((item, index) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.05, ease: [0.22, 1, 0.36, 1] }}
          whileHover={{ y: -2 }}
        >
          <div
            className="dashboard-kpi h-full"
            style={{ "--kpi-surface": item.bg } as React.CSSProperties}
          >
            <div className="text-[14px] font-medium text-[var(--text-secondary)]">{item.label}</div>
            <div className="mt-auto flex items-end justify-between gap-4">
              <div className="dashboard-value break-all text-[22px] sm:text-[24px]">{item.value}</div>
              <div className="flex items-center gap-1 text-[12px] font-medium text-[var(--text-secondary)]">
                <span className="max-w-[108px] truncate text-right">{item.change}</span>
                {item.trend === "up" ? (
                  <TrendingUp className="size-3.5 shrink-0 text-[var(--foreground)]" />
                ) : (
                  <TrendingDown className="size-3.5 shrink-0 text-[var(--foreground)]" />
                )}
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
