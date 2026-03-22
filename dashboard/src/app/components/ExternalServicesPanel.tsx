import { BarChart, Bar, ResponsiveContainer, Tooltip } from "recharts";
import { Layers3, Radio } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, asRecord, formatMs, statusLabel, statusTone, stringValue } from "../selectors";
import type { ArchitectureNode } from "../types";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle, toneFromStatusTone } from "./console-ui";

function statusClasses(status: string) {
  return toneFromStatusTone(statusTone(status));
}

function metricRows(node: ArchitectureNode) {
  const metrics = asRecord(node.metrics);
  return [
    { label: "Status", value: statusLabel(node.status) },
    { label: "Required", value: node.required ? "yes" : "no" },
    { label: "Summary", value: node.summary || "idle" },
    { label: "Detail", value: node.detail || "n/a" },
    { label: "Latency", value: formatMs(node.latencyMs, "n/a") },
    { label: "Signal", value: stringValue(metrics.recoveryState, stringValue(metrics.taskState, stringValue(metrics.activeCommandType, "n/a"))) },
  ];
}

function ModuleCard({
  node,
  latencyData,
  barColor,
}: {
  node: ArchitectureNode;
  latencyData: { t: number; v: number }[];
  barColor: string;
}) {
  const metrics = metricRows(node);

  return (
    <div className="dashboard-panel-strong p-4 flex-1 min-w-0">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <Radio className="size-3.5 text-[var(--text-faint)]" />
          <span className="text-[12px] font-medium text-[var(--foreground)]">{node.name}</span>
        </div>
        <ConsoleBadge tone={statusClasses(node.status)}>
          {statusLabel(node.status)}
        </ConsoleBadge>
      </div>

      <div className="dashboard-micro mb-2 truncate">{node.summary || "idle"}</div>

      <div className="grid grid-cols-3 gap-1.5 text-[10px] mb-2">
        {metrics.slice(0, 6).map((item) => (
          <div key={item.label} className="dashboard-field !rounded-[16px] !px-2 !py-2">
            <div className="dashboard-eyebrow !text-[10px] !tracking-[0.12em]">{item.label}</div>
            <div className="mt-1 truncate text-[11px] font-medium text-[var(--foreground)]">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="h-[32px]">
        <ResponsiveContainer width="100%" height="100%" minWidth={120} minHeight={32}>
          <BarChart data={latencyData.length > 0 ? latencyData : [{ t: 0, v: Number(node.latencyMs ?? 0) || 0 }]}>
            <Bar dataKey="v" fill={barColor} radius={[2, 2, 0, 0]} />
            <Tooltip
              contentStyle={{ background: "rgba(255,251,246,0.98)", border: "1px solid rgba(123,102,79,0.12)", borderRadius: 8, fontSize: 10 }}
              labelStyle={{ display: "none" }}
              formatter={(value: number) => [`${value}ms`, "Latency"]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="dashboard-micro mt-2">latency mirror {formatMs(node.latencyMs, "n/a")}</div>
    </div>
  );
}

export function ExternalServicesPanel() {
  const { history, state } = useDashboard();
  const cards: Array<{ node: ArchitectureNode; color: string; latencyData: { t: number; v: number }[] }> = [
    { node: architectureNode(state, "mainControlServer"), color: "#C6D7E3", latencyData: [] },
    { node: architectureNode(state, "gateway"), color: "#B9CCDA", latencyData: [] },
    { node: architectureModule(state, "s2"), color: "#D4D1EC", latencyData: history.s2Latency },
    { node: architectureModule(state, "nav"), color: "#BED7C5", latencyData: history.navLatency },
    { node: architectureModule(state, "perception"), color: "#E8CCA2", latencyData: [] },
    { node: architectureModule(state, "memory"), color: "#DED1E9", latencyData: [] },
    { node: architectureModule(state, "locomotion"), color: "#E9C2B2", latencyData: [] },
    { node: architectureModule(state, "telemetry"), color: "#D3E0E8", latencyData: [] },
  ];

  return (
    <ConsolePanel>
      <ConsoleSectionTitle
        icon={Layers3}
        eyebrow="health matrix"
        title="External Services"
        description="robot gateway, main control server, and runtime modules arranged on one health surface"
        className="mb-4"
      />
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {cards.map((card) => (
          <ModuleCard
            key={card.node.name}
            node={card.node}
            latencyData={card.latencyData}
            barColor={card.color}
          />
        ))}
      </div>
    </ConsolePanel>
  );
}
