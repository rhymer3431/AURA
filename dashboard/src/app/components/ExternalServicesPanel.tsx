import { BarChart, Bar, ResponsiveContainer, Tooltip } from "recharts";
import { Radio } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, formatMs, processByName, serviceSnapshot, statusLabel, statusTone, stringValue } from "../selectors";

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

function ServiceCard({
  name,
  url,
  status,
  avgLatency,
  latencyData,
  barColor,
  meta,
}: {
  name: string;
  url: string;
  status: string;
  avgLatency: string;
  latencyData: { t: number; v: number }[];
  barColor: string;
  meta: Array<{ label: string; value: string }>;
}) {
  return (
    <div className="bg-white rounded-2xl p-4 flex-1 min-w-0">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <Radio className="size-3.5 text-black/30" />
          <span className="text-[12px] font-medium text-black">{name}</span>
        </div>
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border ${statusClasses(status)}`}>
          <span className={`size-1.5 rounded-full ${statusTone(status) === "green" ? "bg-emerald-500" : statusTone(status) === "amber" ? "bg-amber-500" : statusTone(status) === "red" ? "bg-red-500" : "bg-slate-500"}`} />
          {statusLabel(status)}
        </span>
      </div>

      <div className="text-[10px] text-black/30 mb-2 truncate">{url}</div>

      <div className="grid grid-cols-4 gap-1.5 text-[10px] mb-2">
        {meta.map((item) => (
          <div key={item.label} className="bg-black/[0.02] rounded-lg px-2 py-1.5">
            <div className="text-black/30">{item.label}</div>
            <div className="text-black/80 font-medium">{item.value}</div>
          </div>
        ))}
      </div>

      <div className="h-[32px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={latencyData.length > 0 ? latencyData : [{ t: 0, v: 0 }]}>
            <Bar dataKey="v" fill={barColor} radius={[2, 2, 0, 0]} />
            <Tooltip
              contentStyle={{ background: "#fff", border: "1px solid rgba(0,0,0,0.1)", borderRadius: 8, fontSize: 10 }}
              labelStyle={{ display: "none" }}
              formatter={(value: number) => [`${value}ms`, "Latency"]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-[10px] text-black/40">avg latency {avgLatency}</div>
    </div>
  );
}

export function ExternalServicesPanel() {
  const { history, state } = useDashboard();
  const navdp = serviceSnapshot(state, "navdp");
  const dual = serviceSnapshot(state, "dual");
  const system2 = processByName(state, "system2");
  const system2Info = asRecord(state?.services.system2);

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6">
      <div className="flex items-center gap-2 mb-4">
        <h3 className="text-[15px] font-semibold text-black">Planning Backends</h3>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <ServiceCard
          name="NavDP Backend"
          url={stringValue(navdp.healthUrl, "http://127.0.0.1:8888/health")}
          status={stringValue(navdp.status, "unknown")}
          avgLatency={formatMs(navdp.latencyMs, "n/a")}
          latencyData={history.navdpLatency}
          barColor="#A8C5DA"
          meta={[
            { label: "Status", value: statusLabel(stringValue(navdp.status, "unknown")) },
            { label: "Health", value: Object.keys(asRecord(navdp.health)).length > 0 ? "ok" : "n/a" },
            { label: "Debug", value: Object.keys(asRecord(navdp.debug)).length > 0 ? "ready" : "n/a" },
            { label: "Latency", value: formatMs(navdp.latencyMs, "n/a") },
          ]}
        />
        <ServiceCard
          name="Dual Coordinator"
          url={stringValue(dual.healthUrl, "http://127.0.0.1:8890/health")}
          status={stringValue(dual.status, "inactive")}
          avgLatency={formatMs(dual.latencyMs, "n/a")}
          latencyData={history.dualLatency}
          barColor="#C6C7F8"
          meta={[
            { label: "Status", value: statusLabel(stringValue(dual.status, "inactive")) },
            { label: "Health", value: Object.keys(asRecord(dual.health)).length > 0 ? "ok" : "n/a" },
            { label: "Debug", value: Object.keys(asRecord(dual.debug)).length > 0 ? "ready" : "n/a" },
            { label: "Latency", value: formatMs(dual.latencyMs, "n/a") },
          ]}
        />
        <div className="bg-white rounded-2xl p-4 flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <Radio className="size-3.5 text-black/30" />
              <span className="text-[12px] font-medium text-black">System2 Planner</span>
            </div>
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border ${statusClasses(stringValue(system2?.state, "inactive"))}`}>
              {statusLabel(stringValue(system2?.state, "inactive"))}
            </span>
          </div>
          <div className="text-[10px] text-black/30 mb-3 truncate">{stringValue(system2?.healthUrl, "http://127.0.0.1:8080")}</div>
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <div className="bg-black/[0.02] rounded-lg px-2 py-2">
              <div className="text-black/30">PID</div>
              <div className="text-black/80 font-medium">{system2?.pid ?? "n/a"}</div>
            </div>
            <div className="bg-black/[0.02] rounded-lg px-2 py-2">
              <div className="text-black/30">Required</div>
              <div className="text-black/80 font-medium">{system2?.required ? "yes" : "no"}</div>
            </div>
            <div className="bg-black/[0.02] rounded-lg px-2 py-2 col-span-2">
              <div className="text-black/30">Logs</div>
              <div className="text-black/80 font-medium truncate">{stringValue(system2Info.stdoutLog, system2?.stdoutLog ?? "n/a")}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
