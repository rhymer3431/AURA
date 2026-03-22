import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip,
} from "recharts";
import { Navigation, AlertTriangle } from "lucide-react";

import { useDashboard } from "../state";
import { architectureNode, asRecord, formatMeters, formatRadians, formatSeconds, stringValue } from "../selectors";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function Badge({ color, children }: { color: "green" | "blue" | "amber"; children: React.ReactNode }) {
  const tones = {
    green: "emerald",
    blue: "cyan",
    amber: "amber",
  };
  return (
    <ConsoleBadge tone={tones[color] as "emerald" | "cyan" | "amber"}>{children}</ConsoleBadge>
  );
}

export function NavigationControlPanel() {
  const { history, state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const actionStatus = asRecord(runtime.actionStatus);
  const controlServer = architectureNode(state, "mainControlServer");
  const serverMetrics = asRecord(controlServer.metrics);
  const currentPhase = stringValue(runtime.plannerControlMode, "idle");
  const executionMode = stringValue(runtime.executionMode || runtime.modes?.executionMode, "IDLE");
  const recoveryState = stringValue(runtime.recoveryState, "NORMAL");
  const recoveryReason = stringValue(runtime.recoveryReason, "clear");
  const phaseTone =
    currentPhase === "trajectory"
      ? "green"
      : currentPhase === "yaw_delta" || executionMode === "EXPLORE"
        ? "blue"
        : "amber";

  const metrics = [
    { label: "Plan Ver.", value: `v${Number(runtime.planVersion ?? 0)}` },
    { label: "Goal Ver.", value: `v${Number(runtime.goalVersion ?? 0)}` },
    { label: "Traj Ver.", value: `v${Number(runtime.trajVersion ?? 0)}` },
    { label: "Stale", value: formatSeconds(runtime.staleSec, "n/a") },
    { label: "Goal Dist", value: formatMeters(runtime.goalDistanceM, "n/a") },
    { label: "Yaw Err", value: formatRadians(runtime.plannerYawDeltaRad ?? runtime.yawErrorRad, "n/a") },
    { label: "Recovery", value: recoveryState },
  ];

  return (
    <ConsolePanel className="flex-1 min-w-0">
      <ConsoleSectionTitle
        icon={Navigation}
        eyebrow="planner trace"
        title="Planner & Control"
        description="planning context, command arbitration, and recovery state projected from the main control server snapshot"
        action={<Badge color={phaseTone}>{currentPhase}</Badge>}
        className="mb-3.5"
      />

      <div className="dashboard-inset mb-3.5 p-3.5">
        <div className="grid grid-cols-2 gap-3 text-[11px] md:grid-cols-5">
          <div>
            <div className="dashboard-eyebrow mb-1">execution_mode</div>
            <div className="font-medium text-[var(--foreground)]">{executionMode}</div>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">task_state</div>
            <div className="font-medium text-[var(--foreground)]">{stringValue(serverMetrics.taskState, "idle")}</div>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">planning_phase</div>
            <div className="font-medium text-[var(--signal-emerald)]">{currentPhase}</div>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">control_mode</div>
            <div className="font-medium text-[var(--foreground)]">{stringValue(runtime.plannerControlMode, "idle")}</div>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">ActionStatus</div>
            <div className="font-medium text-[var(--signal-emerald)]">{stringValue(actionStatus.state, "n/a")}</div>
          </div>
        </div>
      </div>

      <div className="mb-3.5 grid grid-cols-3 gap-2.5 md:grid-cols-7">
        {metrics.map((metric) => (
          <div key={metric.label} className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">{metric.label}</div>
            <div className="dashboard-mono text-[16px] font-medium text-[var(--foreground)] leading-tight">{metric.value}</div>
          </div>
        ))}
      </div>

      <div className="dashboard-inset mb-4 flex flex-wrap items-center gap-2 px-3 py-2">
        <span className="dashboard-eyebrow">Arbitration</span>
        <Badge color={phaseTone}>{stringValue(runtime.plannerControlMode, "idle")}</Badge>
        <span className="dashboard-micro">instruction: {stringValue(runtime.activeInstruction, "none") || "none"}</span>
        <div className="flex-1" />
        <div className="flex items-center gap-1.5 dashboard-micro">
          <AlertTriangle className="size-3.5" />
          status: <span className="text-[var(--foreground)]">{stringValue(actionStatus.reason, "clear")}</span>
        </div>
        <div className="flex items-center gap-1.5 dashboard-micro">
          recovery: <span className="text-[var(--foreground)]">{recoveryState}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="dashboard-panel-strong p-3.5">
          <div className="dashboard-title mb-3 text-[13px]">Trajectory Freshness (s)</div>
          <div className="h-[108px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history.stale}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(123,102,79,0.12)" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgba(123,102,79,0.48)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <YAxis tick={{ fontSize: 10, fill: "rgba(123,102,79,0.48)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <Tooltip contentStyle={{ background: "rgba(255,251,246,0.98)", border: "1px solid rgba(123,102,79,0.12)", borderRadius: 8, fontSize: 11 }} />
                <Line type="monotone" dataKey="v" stroke="#c58d4a" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="dashboard-panel-strong p-3.5">
          <div className="dashboard-title mb-3 text-[13px]">Goal Distance (m)</div>
          <div className="h-[108px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history.goalDistance}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(123,102,79,0.12)" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgba(123,102,79,0.48)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <YAxis tick={{ fontSize: 10, fill: "rgba(123,102,79,0.48)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <Tooltip contentStyle={{ background: "rgba(255,251,246,0.98)", border: "1px solid rgba(123,102,79,0.12)", borderRadius: 8, fontSize: 11 }} />
                <Line type="monotone" dataKey="v" stroke="#7699ae" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-strong mt-4 p-3.5">
        <div className="dashboard-eyebrow mb-2">Main Control Server Snapshot</div>
        <div className="grid grid-cols-1 gap-2.5 text-[11px] md:grid-cols-4">
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">Execution Mode</div>
            <div className="font-medium text-[var(--foreground)]">{executionMode}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">Active Command</div>
            <div className="font-medium text-[var(--foreground)]">{stringValue(runtime.activeCommandType, "none")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">Status Reason</div>
            <div className="font-medium text-[var(--foreground)]">{stringValue(actionStatus.reason, "clear")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">Recovery Reason</div>
            <div className="font-medium text-[var(--foreground)]">{recoveryReason}</div>
          </div>
        </div>
        <div className="dashboard-micro mt-3">
          {controlServer.summary || "Main control server ready"} · {controlServer.detail || "No active task"}
        </div>
      </div>
    </ConsolePanel>
  );
}
