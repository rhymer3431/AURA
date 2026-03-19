import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip,
} from "recharts";
import { Navigation, AlertTriangle } from "lucide-react";

import { useDashboard } from "../state";
import { architectureNode, asRecord, formatMeters, formatRadians, formatSeconds, stringValue } from "../selectors";

function Badge({ color, children }: { color: "green" | "blue" | "amber"; children: React.ReactNode }) {
  const tones = {
    green: "bg-emerald-50 text-emerald-600 border-emerald-200",
    blue: "bg-sky-50 text-sky-600 border-sky-200",
    amber: "bg-amber-50 text-amber-600 border-amber-200",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] border ${tones[color]}`}>
      {children}
    </span>
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
    <div className="bg-[#F7F9FB] rounded-3xl p-6 flex-1 min-w-0">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Navigation className="size-4 text-black/40" />
          <div>
            <h3 className="text-[15px] font-semibold text-black">Planner & Control</h3>
            <p className="text-[12px] text-black/50 mt-0.5">planning context, command arbitration, recovery state machine을 main control server 기준으로 보여줍니다.</p>
          </div>
        </div>
        <Badge color={phaseTone}>{currentPhase}</Badge>
      </div>

      <div className="bg-black/[0.02] border border-black/[0.06] rounded-xl p-4 mb-4">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-[12px]">
          <div>
            <div className="text-black/50 mb-1 text-[11px]">execution_mode</div>
            <div className="text-black font-medium">{executionMode}</div>
          </div>
          <div>
            <div className="text-black/50 mb-1 text-[11px]">task_state</div>
            <div className="text-black font-medium">{stringValue(serverMetrics.taskState, "idle")}</div>
          </div>
          <div>
            <div className="text-black/50 mb-1 text-[11px]">planning_phase</div>
            <div className="text-emerald-600 font-medium">{currentPhase}</div>
          </div>
          <div>
            <div className="text-black/50 mb-1 text-[11px]">control_mode</div>
            <div className="text-black font-medium">{stringValue(runtime.plannerControlMode, "idle")}</div>
          </div>
          <div>
            <div className="text-black/50 mb-1 text-[11px]">ActionStatus</div>
            <div className="text-emerald-600 font-medium">{stringValue(actionStatus.state, "n/a")}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 md:grid-cols-7 gap-3 mb-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="bg-black/[0.02] border border-black/[0.06] rounded-xl px-3 py-2.5">
            <div className="text-[10px] text-black/40 uppercase tracking-wider mb-0.5">{metric.label}</div>
            <div className="text-[16px] font-medium text-black leading-tight">{metric.value}</div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2.5 mb-5 bg-black/[0.01] border border-black/5 rounded-lg px-3 py-2 flex-wrap">
        <span className="text-[11px] text-black/50 font-medium">Arbitration:</span>
        <Badge color={phaseTone}>{stringValue(runtime.plannerControlMode, "idle")}</Badge>
        <span className="text-[11px] text-black/40">instruction: {stringValue(runtime.activeInstruction, "none") || "none"}</span>
        <div className="flex-1" />
        <div className="flex items-center gap-1.5 text-[11px] text-black/40 font-medium">
          <AlertTriangle className="size-3.5" />
          status: <span className="text-black/60">{stringValue(actionStatus.reason, "clear")}</span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-black/40 font-medium">
          recovery: <span className="text-black/60">{recoveryState}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-white rounded-2xl p-4">
          <div className="text-[12px] font-medium text-black/60 mb-3">Trajectory Freshness (s)</div>
          <div className="h-[120px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history.stale}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgba(0,0,0,0.4)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <YAxis tick={{ fontSize: 10, fill: "rgba(0,0,0,0.4)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid rgba(0,0,0,0.1)", borderRadius: 8, fontSize: 11 }} />
                <Line type="monotone" dataKey="v" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-white rounded-2xl p-4">
          <div className="text-[12px] font-medium text-black/60 mb-3">Goal Distance (m)</div>
          <div className="h-[120px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history.goalDistance}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "rgba(0,0,0,0.4)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <YAxis tick={{ fontSize: 10, fill: "rgba(0,0,0,0.4)" }} axisLine={false} tickLine={false} tickMargin={5} />
                <Tooltip contentStyle={{ background: "#fff", border: "1px solid rgba(0,0,0,0.1)", borderRadius: 8, fontSize: 11 }} />
                <Line type="monotone" dataKey="v" stroke="#0ea5e9" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-5 bg-white rounded-2xl p-4">
        <div className="text-[11px] font-medium text-black/50 mb-2">Main Control Server Snapshot</div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-[11px]">
          <div className="bg-[#F7F9FB] rounded-xl px-3 py-2">
            <div className="text-black/40 mb-1">Execution Mode</div>
            <div className="text-black/80 font-medium">{executionMode}</div>
          </div>
          <div className="bg-[#F7F9FB] rounded-xl px-3 py-2">
            <div className="text-black/40 mb-1">Active Command</div>
            <div className="text-black/80 font-medium">{stringValue(runtime.activeCommandType, "none")}</div>
          </div>
          <div className="bg-[#F7F9FB] rounded-xl px-3 py-2">
            <div className="text-black/40 mb-1">Status Reason</div>
            <div className="text-black/80 font-medium">{stringValue(actionStatus.reason, "clear")}</div>
          </div>
          <div className="bg-[#F7F9FB] rounded-xl px-3 py-2">
            <div className="text-black/40 mb-1">Recovery Reason</div>
            <div className="text-black/80 font-medium">{recoveryReason}</div>
          </div>
        </div>
        <div className="mt-3 text-[11px] text-black/45">
          {controlServer.summary || "Main control server ready"} · {controlServer.detail || "No active task"}
        </div>
      </div>
    </div>
  );
}
