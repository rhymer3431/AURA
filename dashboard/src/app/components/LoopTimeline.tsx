import type { ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ActivitySquare } from "lucide-react";

import { useDashboard } from "../state";
import { ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function MetricChart({
  title,
  data,
  color,
}: {
  title: string;
  data: { t: number; v: number }[];
  color: string;
}) {
  return (
    <div className="dashboard-panel-strong p-3">
      <div className="mb-2 text-[12px] font-semibold text-[var(--foreground)]">{title}</div>
      <div className="h-[104px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(var(--ink-rgb),0.08)" vertical={false} />
            <XAxis dataKey="t" tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={false} tickLine={false} tickMargin={5} />
            <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} axisLine={false} tickLine={false} tickMargin={5} />
            <Tooltip
              contentStyle={{
                background: "var(--surface-strong)",
                border: "1px solid rgba(var(--ink-rgb),0.08)",
                borderRadius: 8,
                fontSize: 11,
              }}
            />
            <Line type="monotone" dataKey="v" stroke={color} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TimelineList({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="dashboard-panel-strong p-3.5">
      <div className="mb-3 text-[12px] font-semibold text-[var(--foreground)]">{title}</div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

export function LoopTimeline() {
  const { history, state } = useDashboard();
  const traceTail = [...(state?.cognitionTrace ?? [])].slice(-6).reverse();
  const transitionsTail = [...(state?.recoveryTransitions ?? [])].slice(-5).reverse();

  return (
    <ConsolePanel className="flex flex-col gap-4">
      <ConsoleSectionTitle
        icon={ActivitySquare}
        eyebrow="time and events"
        title="Loop Timeline"
        description="stale, goal distance, nav/s2 latency와 recovery/event marker를 같은 시간축 주변에 배치합니다."
      />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-4">
        <MetricChart title="Trajectory Freshness (s)" data={history.stale} color="var(--chart-4)" />
        <MetricChart title="Goal Distance (m)" data={history.goalDistance} color="var(--chart-2)" />
        <MetricChart title="Nav Latency (ms)" data={history.navLatency} color="var(--signal-emerald)" />
        <MetricChart title="S2 Latency (ms)" data={history.s2Latency} color="var(--signal-violet)" />
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <TimelineList title="Recovery Transition Markers">
          {transitionsTail.length === 0 ? (
            <div className="text-[11px] text-[var(--text-tertiary)]">No recovery transitions yet.</div>
          ) : (
            transitionsTail.map((item, index) => (
              <div key={`${item.to}-${index}`} className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">transition</div>
                <div className="text-[12px] font-medium text-[var(--foreground)]">
                  {item.from}
                  {" -> "}
                  {item.to}
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-secondary)]">
                  {item.reason || "no reason"} · retry {item.retryCount}
                </div>
              </div>
            ))
          )}
        </TimelineList>

        <TimelineList title="Recent Trace Events">
          {traceTail.length === 0 ? (
            <div className="text-[11px] text-[var(--text-tertiary)]">No frame trace entries yet.</div>
          ) : (
            traceTail.map((item) => (
              <div key={`${item.frameId}-${item.planVersion}-${item.activeCommandType}`} className="dashboard-field">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-[12px] font-medium text-[var(--foreground)]">frame {item.frameId}</div>
                  <div className="dashboard-mono text-[10px] text-[var(--text-tertiary)]">{item.s2DecisionMode || "no_s2"}</div>
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-secondary)]">
                  {item.activeCommandType || "idle"} · {item.recoveryState} · {item.actionReason || item.recoveryReason || "clear"}
                </div>
              </div>
            ))
          )}
        </TimelineList>
      </div>
    </ConsolePanel>
  );
}
