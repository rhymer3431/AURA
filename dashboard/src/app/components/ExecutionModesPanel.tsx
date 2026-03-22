import { Crosshair, Eye, Layers3, ScanLine, SlidersHorizontal } from "lucide-react";

import { useDashboard } from "../state";
import { booleanValue } from "../selectors";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function ModeBadge({
  label,
  enabled,
}: {
  label: string;
  enabled: boolean;
}) {
  return (
    <ConsoleBadge tone={enabled ? "emerald" : "slate"}>{label}</ConsoleBadge>
  );
}

export function ExecutionModesPanel() {
  const { bootstrap, form, state } = useDashboard();
  const liveConfig = state?.session.config;
  const executionModes = bootstrap?.executionModes ?? ["TALK", "NAV", "MEM_NAV", "EXPLORE", "IDLE"];
  const launchModes = bootstrap?.launchModes ?? ["gui", "headless"];
  const locomotionText = [
    `action ${form.locomotionConfig.actionScale}`,
    form.locomotionConfig.onnxDevice,
    `vx ${form.locomotionConfig.cmdMaxVx}`,
    `vy ${form.locomotionConfig.cmdMaxVy}`,
    `wz ${form.locomotionConfig.cmdMaxWz}`,
  ].join(" / ");
  const liveLocomotionText =
    liveConfig == null
      ? "inactive"
      : [
          `action ${liveConfig.locomotionConfig.actionScale.toFixed(3)}`,
          liveConfig.locomotionConfig.onnxDevice,
          `vx ${liveConfig.locomotionConfig.cmdMaxVx}`,
          `vy ${liveConfig.locomotionConfig.cmdMaxVy}`,
          `wz ${liveConfig.locomotionConfig.cmdMaxWz}`,
        ].join(" / ");

  return (
    <ConsolePanel className="flex flex-col gap-5">
      <ConsoleSectionTitle
        icon={SlidersHorizontal}
        eyebrow="profile compare"
        title="Runtime Modes"
        description="draft runtime profile과 현재 활성 세션 설정을 분리해서 보여줍니다."
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="dashboard-panel-strong p-5">
          <div className="dashboard-title mb-4 text-[13px]">Draft Runtime Profile</div>
          <div className="grid grid-cols-2 gap-3 text-[12px]">
            <div className="dashboard-field">
              <div className="dashboard-eyebrow mb-1">Gateway Launch</div>
              <div className="font-medium text-[var(--foreground)]">{form.launchMode}</div>
            </div>
            <div className="dashboard-field">
              <div className="dashboard-eyebrow mb-1">Scene Preset</div>
              <div className="font-medium text-[var(--foreground)]">{form.scenePreset}</div>
            </div>
            <div className="dashboard-field col-span-2">
              <div className="dashboard-eyebrow mb-1">G1 Locomotion Config</div>
              <div className="dashboard-mono text-[12px] text-[var(--foreground)] break-all">{locomotionText}</div>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <ModeBadge label="viewer publish" enabled={form.viewerEnabled} />
            <ModeBadge label="memory store" enabled={form.memoryStore} />
            <ModeBadge label="detection" enabled={form.detectionEnabled} />
          </div>
        </div>

        <div className="dashboard-panel-strong p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="dashboard-title text-[13px]">Active Runtime Session</div>
            <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
              {state?.session.active ? "running" : "idle"}
            </ConsoleBadge>
          </div>
          <div className="grid grid-cols-2 gap-3 text-[12px]">
            <div className="dashboard-field">
              <div className="dashboard-eyebrow mb-1">Execution Mode</div>
              <div className="font-medium text-[var(--foreground)]">{String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE")}</div>
            </div>
            <div className="dashboard-field">
              <div className="dashboard-eyebrow mb-1">Gateway Launch</div>
              <div className="font-medium text-[var(--foreground)]">{liveConfig?.launchMode ?? "inactive"}</div>
            </div>
            <div className="dashboard-field">
              <div className="dashboard-eyebrow mb-1">Scene Preset</div>
              <div className="font-medium text-[var(--foreground)]">{liveConfig?.scenePreset ?? "inactive"}</div>
            </div>
            <div className="dashboard-field col-span-2">
              <div className="dashboard-eyebrow mb-1">G1 Locomotion Config</div>
              <div className="dashboard-mono text-[12px] text-[var(--foreground)] break-all">{liveLocomotionText}</div>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <ModeBadge label="viewer publish" enabled={booleanValue(liveConfig?.viewerEnabled)} />
            <ModeBadge label="memory store" enabled={booleanValue(liveConfig?.memoryStore)} />
            <ModeBadge label="detection" enabled={booleanValue(liveConfig?.detectionEnabled)} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="dashboard-panel-strong p-4">
          <div className="flex items-center gap-2 text-[12px] font-medium text-[var(--text-secondary)] mb-2">
            <Layers3 className="size-4 text-[var(--text-faint)]" />
            Execution Modes
          </div>
          <div className="dashboard-mono text-[12px] text-[var(--text-secondary)] leading-6">{executionModes.join(" / ")}</div>
        </div>
        <div className="dashboard-panel-strong p-4">
          <div className="flex items-center gap-2 text-[12px] font-medium text-[var(--text-secondary)] mb-2">
            <Eye className="size-4 text-[var(--text-faint)]" />
            Gateway Launch Modes
          </div>
          <div className="dashboard-mono text-[12px] text-[var(--text-secondary)] leading-6">{launchModes.join(" / ")}</div>
        </div>
        <div className="dashboard-panel-strong p-4">
          <div className="flex items-center gap-2 text-[12px] font-medium text-[var(--text-secondary)] mb-2">
            <Crosshair className="size-4 text-[var(--text-faint)]" />
            Active Control Surface
          </div>
          <div className="dashboard-mono text-[12px] text-[var(--text-secondary)] leading-6">
            {state?.session.active ? String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE") : "no active session"}
          </div>
        </div>
        <div className="dashboard-panel-strong p-4">
          <div className="flex items-center gap-2 text-[12px] font-medium text-[var(--text-secondary)] mb-2">
            <ScanLine className="size-4 text-[var(--text-faint)]" />
            Detection Gate
          </div>
          <div className="dashboard-mono text-[12px] text-[var(--text-secondary)] leading-6">
            {booleanValue(liveConfig?.detectionEnabled, form.detectionEnabled) ? "enabled" : "disabled"}
          </div>
        </div>
      </div>
    </ConsolePanel>
  );
}
