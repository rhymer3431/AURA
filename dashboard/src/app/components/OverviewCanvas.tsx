import { motion } from "motion/react";
import { Activity, ArrowUpRight, Gauge } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useDashboard } from "../state";
import {
  architectureModule,
  architectureNode,
  asRecord,
  formatMeters,
  formatMs,
  numberValue,
  statusLabel,
  statusTone,
  stringValue,
} from "../selectors";
import { PipelineFlow } from "./PipelineFlow";
import { ProcessesWidget, SensorsWidget } from "./SystemStatusWidgets";
import { RobotViewer } from "./RobotViewer";
import { StatCards } from "./StatCards";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

const statusDotClass: Record<string, string> = {
  green: "bg-[var(--signal-emerald)]",
  amber: "bg-[var(--signal-amber)]",
  red: "bg-[var(--signal-coral)]",
  slate: "bg-[var(--signal-slate)]",
};

function buildTrendSeries(series: Array<{ t: number; v: number }>, targetLength: number) {
  const startIndex = Math.max(0, series.length - targetLength);
  return Array.from({ length: targetLength }, (_, index) => {
    const source = series[startIndex + index];
    return source?.v ?? null;
  });
}

function buildTrendData(history: ReturnType<typeof useDashboard>["history"]) {
  const targetLength = Math.max(
    history.goalDistance.length,
    history.navLatency.length,
    history.s2Latency.length,
    8,
  );
  const goalDistance = buildTrendSeries(history.goalDistance, targetLength);
  const navLatency = buildTrendSeries(history.navLatency, targetLength);
  const s2Latency = buildTrendSeries(history.s2Latency, targetLength);

  return Array.from({ length: targetLength }, (_, index) => ({
    label: `${index + 1}`,
    goalDistance: goalDistance[index],
    navLatency: navLatency[index],
    s2Latency: s2Latency[index],
  }));
}

export function OverviewCanvas() {
  const { history, state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const transport = asRecord(state?.transport);
  const memory = asRecord(state?.memory);
  const chartData = buildTrendData(history);
  const nav = architectureModule(state, "nav");
  const perception = architectureModule(state, "perception");
  const telemetry = architectureModule(state, "telemetry");
  const memoryModule = architectureModule(state, "memory");
  const gateway = architectureNode(state, "gateway");
  const planner = architectureNode(state, "mainControlServer");
  const frameAge = formatMs(transport.frameAgeMs, "n/a");
  const goalDistance = formatMeters(runtime.goalDistanceM, "n/a");
  const planLatency = formatMs(runtime.planLatencyMs ?? runtime.plannerLatencyMs, "n/a");
  const detectionCount = numberValue(asRecord(state?.perception).detectionCount) ?? 0;
  const trackedCount = numberValue(asRecord(state?.perception).trackedDetectionCount) ?? 0;
  const instruction = stringValue(asRecord(memory.scratchpad).instruction, "No active instruction");

  const statusRows = [
    gateway,
    planner,
    nav,
    perception,
    telemetry,
    memoryModule,
  ].map((item) => ({
    name: item.name,
    summary: item.summary || item.detail || "idle",
    tone: statusTone(item.status),
    label: statusLabel(item.status),
  }));

  return (
    <div className="space-y-4">
      <StatCards />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.85fr)_minmax(280px,0.85fr)]">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        >
          <ConsolePanel className="h-full">
            <ConsoleSectionTitle
              icon={Activity}
              eyebrow="Operational signal"
              title="Live trend canvas"
              description="Goal distance, Nav latency, and S2 latency across the latest runtime samples."
              className="mb-4"
            />

            <div className="h-[248px] rounded-[16px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-strong)] px-2 py-2">
              <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={240}>
                <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="rgba(var(--ink-rgb),0.06)" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: "var(--text-tertiary)", fontSize: 11 }}
                  />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: "var(--text-tertiary)", fontSize: 11 }}
                    width={36}
                  />
                  <Tooltip
                    contentStyle={{
                      border: "1px solid rgba(var(--ink-rgb),0.08)",
                      borderRadius: "12px",
                      background: "var(--surface-strong)",
                      boxShadow: "0 12px 24px rgba(var(--ink-rgb),0.08)",
                    }}
                    labelStyle={{ color: "var(--text-tertiary)", fontSize: 11 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="goalDistance"
                    stroke="var(--chart-1)"
                    strokeWidth={1.8}
                    dot={false}
                    activeDot={{ r: 4, fill: "var(--chart-1)" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="navLatency"
                    stroke="var(--chart-2)"
                    strokeWidth={1.8}
                    dot={false}
                    activeDot={{ r: 4, fill: "var(--chart-2)" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="s2Latency"
                    stroke="var(--chart-3)"
                    strokeWidth={1.8}
                    dot={false}
                    activeDot={{ r: 4, fill: "var(--chart-3)" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap gap-1.5">
              <ConsoleBadge tone="slate" dot={false}>Goal distance</ConsoleBadge>
              <ConsoleBadge tone="cyan" dot={false}>Nav latency</ConsoleBadge>
              <ConsoleBadge tone="violet" dot={false}>S2 latency</ConsoleBadge>
            </div>
          </ConsolePanel>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
        >
          <ConsolePanel className="h-full">
            <ConsoleSectionTitle
              icon={Gauge}
              eyebrow="Runtime pulse"
              title="System brief"
              description="High-signal operational context without opening a separate monitoring page."
              className="mb-4"
            />

            <div className="grid grid-cols-2 gap-2.5">
              <div className="dashboard-field bg-[var(--tone-cyan-bg)]">
                <div className="dashboard-eyebrow mb-1">Frame age</div>
                <div className="text-[18px] font-semibold text-[var(--foreground)]">{frameAge}</div>
              </div>
              <div className="dashboard-field bg-[var(--tone-violet-bg)]">
                <div className="dashboard-eyebrow mb-1">Goal dist.</div>
                <div className="text-[18px] font-semibold text-[var(--foreground)]">{goalDistance}</div>
              </div>
              <div className="dashboard-field bg-[var(--surface-strong)]">
                <div className="dashboard-eyebrow mb-1">Plan latency</div>
                <div className="text-[18px] font-semibold text-[var(--foreground)]">{planLatency}</div>
              </div>
              <div className="dashboard-field bg-[var(--surface-strong)]">
                <div className="dashboard-eyebrow mb-1">Detections</div>
                <div className="text-[18px] font-semibold text-[var(--foreground)]">
                  {detectionCount}
                  <span className="ml-2 text-[11px] font-normal text-[var(--text-tertiary)]">
                    tracked {trackedCount}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-3 rounded-[16px] border border-[rgba(var(--ink-rgb),0.05)] bg-[var(--surface-strong)] px-3.5 py-3">
              <div className="dashboard-eyebrow mb-2">Current instruction</div>
              <p className="text-[13px] leading-5 text-[var(--text-secondary)]">{instruction}</p>
            </div>

            <div className="mt-3 space-y-2">
              {statusRows.map((item, index) => (
                <motion.div
                  key={item.name}
                  className="flex items-start justify-between gap-3 rounded-[16px] border border-[rgba(var(--ink-rgb),0.05)] bg-[var(--surface-strong)] px-3.5 py-2.5"
                  initial={{ opacity: 0, x: 14 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.28, delay: 0.12 + index * 0.04 }}
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`size-2 rounded-full ${statusDotClass[item.tone] ?? statusDotClass.slate}`} />
                      <div className="truncate text-[13px] font-semibold text-[var(--foreground)]">{item.name}</div>
                    </div>
                    <div className="mt-1 truncate text-[11px] text-[var(--text-tertiary)]">{item.summary}</div>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] text-[var(--text-tertiary)]">
                    {item.label}
                    <ArrowUpRight className="size-3.5" />
                  </div>
                </motion.div>
              ))}
            </div>
          </ConsolePanel>
        </motion.div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(300px,0.86fr)]">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
        >
          <RobotViewer />
        </motion.div>

        <motion.div
          className="space-y-4"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.16, ease: [0.22, 1, 0.36, 1] }}
        >
          <ProcessesWidget />
          <SensorsWidget />
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
      >
        <PipelineFlow />
      </motion.div>
    </div>
  );
}
