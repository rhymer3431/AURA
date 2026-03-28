import type { ReactNode } from "react";
import { ShieldCheck, RadioTower, TimerReset } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, booleanValue, formatMs, stringValue } from "../selectors";
import { ConsoleBadge } from "./console-ui";

function StatusCell({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-[104px] rounded-[14px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-2)] px-2.5 py-2">
      <div className="dashboard-eyebrow mb-1">{label}</div>
      <div className="text-[12px] font-medium text-[var(--foreground)]">{value}</div>
    </div>
  );
}

export function TopStatusStrip() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const transport = asRecord(state?.transport);
  const sessionConfig = state?.session.config;

  return (
    <div className="dashboard-panel flex flex-wrap items-center gap-2 px-3 py-2.5">
      <div className="flex items-center gap-2.5 pr-1">
        <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
          {state?.session.active ? "session active" : "session idle"}
        </ConsoleBadge>
        <ConsoleBadge tone={booleanValue(transport.peerActive) ? "cyan" : "slate"}>
          peer {booleanValue(transport.peerActive) ? "connected" : "inactive"}
        </ConsoleBadge>
      </div>
      <StatusCell label="mode" value={String(runtime.executionMode ?? runtime.modes?.executionMode ?? "IDLE")} />
      <StatusCell label="launch" value={sessionConfig?.launchMode ?? "inactive"} />
      <StatusCell label="task" value={stringValue(runtime.activeInstruction, "idle") || "idle"} />
      <StatusCell label="frame age" value={formatMs(state?.latencyBreakdown.frameAgeMs, "n/a")} />
      <StatusCell label="recovery" value={stringValue(runtime.recoveryState, "NORMAL")} />
      <StatusCell label="action reason" value={stringValue(asRecord(runtime.actionStatus).reason, "clear") || "clear"} />
      <StatusCell
        label="viewer / peer"
        value={
          <span className="inline-flex items-center gap-1.5">
            <RadioTower className="size-3.5 text-[var(--text-tertiary)]" />
            {booleanValue(transport.viewerEnabled) ? "viewer on" : "viewer off"} / {booleanValue(transport.peerActive) ? "peer up" : "peer down"}
          </span>
        }
      />
      <StatusCell
        label="safe stop"
        value={
          <span className="inline-flex items-center gap-1.5">
            {booleanValue(transport.safeStop) ? (
              <ShieldCheck className="size-3.5 text-[var(--signal-coral)]" />
            ) : (
              <TimerReset className="size-3.5 text-[var(--signal-emerald)]" />
            )}
            {booleanValue(transport.safeStop) ? "armed" : "clear"}
          </span>
        }
      />
    </div>
  );
}
