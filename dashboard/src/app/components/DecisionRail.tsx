import type { ReactNode } from "react";
import { Compass, Gauge, Sparkles, TriangleAlert } from "lucide-react";

import { useDashboard } from "../state";
import { asArray, asRecord, formatMeters, formatMs, formatRadians, stringValue } from "../selectors";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle, toneFromStatusTone } from "./console-ui";
import { buildLoopStages, type LoopStageId } from "./liveLoopStages";
import { statusTone } from "../selectors";

function formatVector(value: unknown): string {
  if (!Array.isArray(value) || value.length < 3) {
    return "n/a";
  }
  const [vx, vy, wz] = value;
  if (
    typeof vx !== "number"
    || typeof vy !== "number"
    || typeof wz !== "number"
    || Number.isNaN(vx)
    || Number.isNaN(vy)
    || Number.isNaN(wz)
  ) {
    return "n/a";
  }
  return `[${vx.toFixed(2)}, ${vy.toFixed(2)}, ${wz.toFixed(2)}]`;
}

function Block({
  icon: Icon,
  title,
  badge,
  children,
}: {
  icon: typeof Sparkles;
  title: string;
  badge?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="dashboard-panel-strong p-3.5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Icon className="size-4 text-[var(--text-tertiary)]" />
          <div className="text-[13px] font-semibold text-[var(--foreground)]">{title}</div>
        </div>
        {badge}
      </div>
      {children}
    </div>
  );
}

export function DecisionRail({
  selectedStageId,
}: {
  selectedStageId: LoopStageId;
}) {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const actionStatus = asRecord(runtime.actionStatus);
  const system2Output = state?.services.system2?.output ?? null;
  const stages = buildLoopStages(state);
  const selectedStage = stages.find((stage) => stage.id === selectedStageId) ?? stages[0];
  const trajectory = asArray(runtime.navTrajectoryWorld).slice(0, 4);

  return (
    <ConsolePanel className="flex h-full flex-col gap-4">
      <ConsoleSectionTitle
        icon={Sparkles}
        eyebrow="decision explanation"
        title="Decision Rail"
        description="현재 stage와 S2, motion, recovery 해석을 한 열에서 설명합니다."
        action={<ConsoleBadge tone={toneFromStatusTone(statusTone(selectedStage.status))}>{selectedStage.label}</ConsoleBadge>}
      />

      <div className="dashboard-inset p-3">
        <div className="dashboard-eyebrow mb-1">selected stage</div>
        <div className="text-[14px] font-semibold text-[var(--foreground)]">{selectedStage.summary}</div>
        <div className="mt-1 text-[12px] text-[var(--text-secondary)]">{selectedStage.output}</div>
      </div>

      <Block
        icon={Sparkles}
        title="S2 Decision"
        badge={<ConsoleBadge tone={system2Output ? "emerald" : "amber"}>{system2Output?.decisionMode || "awaiting"}</ConsoleBadge>}
      >
        <div className="dashboard-field dashboard-mono text-[12px] leading-relaxed break-words">
          {system2Output?.rawText || "no S2 output yet"}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">latency</div>
            <div className="dashboard-mono text-[var(--foreground)]">{formatMs(state?.latencyBreakdown.s2LatencyMs, "n/a")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">needs requery</div>
            <div className="text-[var(--foreground)]">{system2Output?.needsRequery ? "yes" : "no"}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">history frames</div>
            <div className="dashboard-mono text-[var(--foreground)]">
              {system2Output?.historyFrameIds?.length ? system2Output.historyFrameIds.join(", ") : "n/a"}
            </div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">instruction</div>
            <div className="text-[var(--foreground)] break-words">{system2Output?.instruction || stringValue(runtime.activeInstruction, "n/a")}</div>
          </div>
        </div>
      </Block>

      <Block
        icon={Compass}
        title="Motion Decision"
        badge={<ConsoleBadge tone="cyan">{stringValue(runtime.activeCommandType, "idle") || "idle"}</ConsoleBadge>}
      >
        <div className="dashboard-field dashboard-mono text-[12px] text-[var(--foreground)]">{formatVector(runtime.commandVector)}</div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">goal distance</div>
            <div className="text-[var(--foreground)]">{formatMeters(runtime.goalDistanceM, "n/a")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">yaw error</div>
            <div className="text-[var(--foreground)]">{formatRadians(runtime.yawErrorRad ?? runtime.plannerYawDeltaRad, "n/a")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">command speed</div>
            <div className="text-[var(--foreground)]">{formatMeters(runtime.commandSpeedMps, "n/a")}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">traj preview</div>
            <div className="dashboard-mono text-[var(--foreground)]">
              {trajectory.length > 0 ? trajectory.map((point) => `(${point[0]}, ${point[1]})`).join(" -> ") : "n/a"}
            </div>
          </div>
        </div>
      </Block>

      <Block
        icon={Gauge}
        title="Resolver / Recovery"
        badge={<ConsoleBadge tone={stringValue(runtime.recoveryState, "NORMAL") === "NORMAL" ? "emerald" : "amber"}>{stringValue(runtime.recoveryState, "NORMAL")}</ConsoleBadge>}
      >
        <div className="grid grid-cols-1 gap-2 text-[11px]">
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">action status</div>
            <div className="text-[var(--foreground)]">{stringValue(actionStatus.state, "n/a") || "n/a"}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">action reason</div>
            <div className="text-[var(--foreground)] break-words">{stringValue(actionStatus.reason, "clear") || "clear"}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">recovery reason</div>
            <div className="text-[var(--foreground)] break-words">{stringValue(runtime.recoveryReason, "clear") || "clear"}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">retry / backoff</div>
            <div className="text-[var(--foreground)]">
              {String(runtime.recoveryRetryCount ?? 0)} / {String(runtime.recoveryBackoffUntilNs ?? 0)}
            </div>
          </div>
        </div>

        {state?.recoveryTransitions.length ? (
          <div className="mt-3 rounded-[16px] border border-[var(--tone-amber-border)] bg-[var(--tone-amber-bg)] px-3 py-2 text-[11px] text-[var(--tone-amber-fg)]">
            <div className="mb-1 flex items-center gap-2 font-medium">
              <TriangleAlert className="size-3.5" />
              latest transition
            </div>
            <div>
              {state.recoveryTransitions[state.recoveryTransitions.length - 1].from}
              {" -> "}
              {state.recoveryTransitions[state.recoveryTransitions.length - 1].to}
              {" · "}
              {state.recoveryTransitions[state.recoveryTransitions.length - 1].reason}
            </div>
          </div>
        ) : null}
      </Block>
    </ConsolePanel>
  );
}
