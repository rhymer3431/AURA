import { ArrowRight, Cpu } from "lucide-react";

import { buildLoopStages, type LoopStageId } from "./liveLoopStages";
import { useDashboard } from "../state";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle, toneFromStatusTone } from "./console-ui";
import { statusTone } from "../selectors";

export function CognitionLoopLane({
  selectedStageId,
  onSelectStageId,
}: {
  selectedStageId: LoopStageId;
  onSelectStageId: (value: LoopStageId) => void;
}) {
  const { state } = useDashboard();
  const stages = buildLoopStages(state);

  return (
    <ConsolePanel className="flex h-full flex-col">
      <ConsoleSectionTitle
        icon={Cpu}
        eyebrow="closed loop trace"
        title="Cognition Loop Lane"
        description="gateway부터 command resolver까지 같은 frame 기준으로 단계별 상태와 출력 요약을 나란히 봅니다."
        className="mb-4"
      />

      <div className="flex flex-col gap-3">
        {stages.map((stage, index) => {
          const active = stage.id === selectedStageId;
          return (
            <div key={stage.id} className="flex items-stretch gap-3">
              <button
                type="button"
                onClick={() => onSelectStageId(stage.id)}
                className={`flex-1 rounded-[20px] border px-4 py-3 text-left transition-colors ${
                  active
                    ? "border-[var(--tone-cyan-border)] bg-[var(--tone-cyan-bg)]"
                    : "border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-2)] hover:bg-[var(--surface-strong)]"
                }`}
              >
                <div className="mb-2 flex items-start justify-between gap-3">
                  <div>
                    <div className="dashboard-eyebrow mb-1">{stage.label}</div>
                    <div className="text-[14px] font-semibold text-[var(--foreground)]">{stage.summary}</div>
                  </div>
                  <ConsoleBadge tone={toneFromStatusTone(statusTone(stage.status))}>{stage.status.replace(/_/g, " ")}</ConsoleBadge>
                </div>
                <div className="mb-2 text-[12px] text-[var(--text-secondary)]">{stage.output}</div>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="dashboard-field">
                    <div className="dashboard-eyebrow mb-1">latency</div>
                    <div className="dashboard-mono text-[var(--foreground)]">{stage.latency}</div>
                  </div>
                  <div className="dashboard-field">
                    <div className="dashboard-eyebrow mb-1">focus</div>
                    <div className="text-[var(--foreground)]">{active ? "selected" : "inspect"}</div>
                  </div>
                </div>
              </button>

              {index < stages.length - 1 ? (
                <div className="hidden w-8 items-center justify-center lg:flex">
                  <ArrowRight className="size-4 text-[var(--text-faint)]" />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </ConsolePanel>
  );
}
