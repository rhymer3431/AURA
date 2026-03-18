import { BarChart, Bar, ResponsiveContainer, Tooltip } from "recharts";
import { Layers3, Radio } from "lucide-react";

import { useDashboard } from "../state";
import { architectureModule, architectureNode, asRecord, formatMs, statusLabel, statusTone, stringValue } from "../selectors";
import type { ArchitectureNode } from "../types";

function statusClasses(status: string) {
  if (statusTone(status) === "green") {
    return "bg-emerald-50 text-emerald-600 border-emerald-200";
  }
  if (statusTone(status) === "amber") {
    return "bg-amber-50 text-amber-600 border-amber-200";
  }
  if (statusTone(status) === "red") {
    return "bg-red-50 text-red-600 border-red-200";
  }
  return "bg-slate-50 text-slate-600 border-slate-200";
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
    <div className="bg-white rounded-2xl p-4 flex-1 min-w-0">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <Radio className="size-3.5 text-black/30" />
          <span className="text-[12px] font-medium text-black">{node.name}</span>
        </div>
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border ${statusClasses(node.status)}`}>
          <span
            className={`size-1.5 rounded-full ${
              statusTone(node.status) === "green"
                ? "bg-emerald-500"
                : statusTone(node.status) === "amber"
                  ? "bg-amber-500"
                  : statusTone(node.status) === "red"
                    ? "bg-red-500"
                    : "bg-slate-500"
            }`}
          />
          {statusLabel(node.status)}
        </span>
      </div>

      <div className="text-[10px] text-black/30 mb-2 truncate">{node.summary || "idle"}</div>

      <div className="grid grid-cols-3 gap-1.5 text-[10px] mb-2">
        {metrics.slice(0, 6).map((item) => (
          <div key={item.label} className="bg-black/[0.02] rounded-lg px-2 py-1.5">
            <div className="text-black/30">{item.label}</div>
            <div className="text-black/80 font-medium truncate">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="h-[32px]">
        <ResponsiveContainer width="100%" height="100%" minWidth={120} minHeight={32}>
          <BarChart data={latencyData.length > 0 ? latencyData : [{ t: 0, v: Number(node.latencyMs ?? 0) || 0 }]}>
            <Bar dataKey="v" fill={barColor} radius={[2, 2, 0, 0]} />
            <Tooltip
              contentStyle={{ background: "#fff", border: "1px solid rgba(0,0,0,0.1)", borderRadius: 8, fontSize: 10 }}
              labelStyle={{ display: "none" }}
              formatter={(value: number) => [`${value}ms`, "Latency"]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-[10px] text-black/40">latency mirror {formatMs(node.latencyMs, "n/a")}</div>
    </div>
  );
}

export function ExternalServicesPanel() {
  const { history, state } = useDashboard();
  const cards: Array<{ node: ArchitectureNode; color: string; latencyData: { t: number; v: number }[] }> = [
    { node: architectureNode(state, "mainControlServer"), color: "#BFDDF6", latencyData: [] },
    { node: architectureNode(state, "gateway"), color: "#A8C5DA", latencyData: [] },
    { node: architectureModule(state, "s2"), color: "#C6C7F8", latencyData: history.s2Latency },
    { node: architectureModule(state, "nav"), color: "#A7E6D7", latencyData: history.navLatency },
    { node: architectureModule(state, "perception"), color: "#F8D6A3", latencyData: [] },
    { node: architectureModule(state, "memory"), color: "#E8D1FF", latencyData: [] },
    { node: architectureModule(state, "locomotion"), color: "#FFC9B8", latencyData: [] },
    { node: architectureModule(state, "telemetry"), color: "#D4E7F7", latencyData: [] },
  ];

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <Layers3 className="size-4 text-black/40" />
        <div>
          <h3 className="text-[15px] font-semibold text-black">Module Health</h3>
          <p className="text-[12px] text-black/50 mt-0.5">robot gateway와 main control server를 포함한 runtime modules 상태를 world state 기준으로 정렬해 보여줍니다.</p>
        </div>
      </div>
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
    </div>
  );
}
