import { Gauge, SlidersHorizontal } from "lucide-react";

import { useDashboard } from "../state";
import { booleanValue, stringValue } from "../selectors";
import { ControlStrip } from "./ControlStrip";
import { ExecutionModesPanel } from "./ExecutionModesPanel";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function SessionSnapshotPanel() {
  const { form, state } = useDashboard();
  const liveConfig = state?.session.config;
  const runtimeMode = String(state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE");
  const lastEvent = state?.session.lastEvent?.message ?? "no recent event";

  return (
    <ConsolePanel className="flex h-full flex-col gap-4">
      <ConsoleSectionTitle
        icon={Gauge}
        eyebrow="current session"
        title="Session Snapshot"
        description="현재 running state, live config, and gating toggles를 상단에서 바로 확인합니다."
        action={
          <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
            {state?.session.active ? "running" : "idle"}
          </ConsoleBadge>
        }
      />

      <div className="grid grid-cols-2 gap-2.5 text-[11px]">
        <div className="dashboard-field">
          <div className="dashboard-eyebrow mb-1">execution mode</div>
          <div className="font-medium text-[var(--foreground)]">{runtimeMode}</div>
        </div>
        <div className="dashboard-field">
          <div className="dashboard-eyebrow mb-1">launch mode</div>
          <div className="font-medium text-[var(--foreground)]">{liveConfig?.launchMode ?? form.launchMode}</div>
        </div>
        <div className="dashboard-field">
          <div className="dashboard-eyebrow mb-1">scene preset</div>
          <div className="font-medium text-[var(--foreground)]">{liveConfig?.scenePreset ?? form.scenePreset}</div>
        </div>
        <div className="dashboard-field">
          <div className="dashboard-eyebrow mb-1">viewer / peer</div>
          <div className="font-medium text-[var(--foreground)]">
            {booleanValue(state?.transport.viewerEnabled, form.viewerEnabled) ? "viewer on" : "viewer off"} /{" "}
            {booleanValue(state?.transport.peerActive) ? "peer up" : "peer down"}
          </div>
        </div>
      </div>

      <div className="dashboard-field">
        <div className="dashboard-eyebrow mb-1">locomotion profile</div>
        <div className="dashboard-mono text-[12px] text-[var(--foreground)] break-all">
          action {liveConfig?.locomotionConfig.actionScale ?? form.locomotionConfig.actionScale} /{" "}
          {liveConfig?.locomotionConfig.onnxDevice ?? form.locomotionConfig.onnxDevice} / vx{" "}
          {liveConfig?.locomotionConfig.cmdMaxVx ?? form.locomotionConfig.cmdMaxVx} / vy{" "}
          {liveConfig?.locomotionConfig.cmdMaxVy ?? form.locomotionConfig.cmdMaxVy} / wz{" "}
          {liveConfig?.locomotionConfig.cmdMaxWz ?? form.locomotionConfig.cmdMaxWz}
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <ConsoleBadge tone={booleanValue(liveConfig?.viewerEnabled, form.viewerEnabled) ? "emerald" : "slate"}>viewer publish</ConsoleBadge>
        <ConsoleBadge tone={booleanValue(liveConfig?.memoryStore, form.memoryStore) ? "emerald" : "slate"}>memory store</ConsoleBadge>
        <ConsoleBadge tone={booleanValue(liveConfig?.detectionEnabled, form.detectionEnabled) ? "emerald" : "slate"}>detection</ConsoleBadge>
        <ConsoleBadge tone={booleanValue(state?.transport.safeStop) ? "amber" : "cyan"}>
          safe stop {booleanValue(state?.transport.safeStop) ? "armed" : "clear"}
        </ConsoleBadge>
      </div>

      <div className="dashboard-inset px-3 py-2.5">
        <div className="dashboard-eyebrow mb-1">last event</div>
        <div className="text-[12px] text-[var(--foreground)] break-words">{stringValue(lastEvent, "no recent event")}</div>
      </div>
    </ConsolePanel>
  );
}

export function SessionConfigWorkspace() {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12 xl:items-start">
        <div className="xl:col-span-8">
          <ControlStrip />
        </div>
        <div className="xl:col-span-4">
          <SessionSnapshotPanel />
        </div>
      </div>

      <div className="space-y-4">
        <div className="px-1">
          <ConsoleSectionTitle
            icon={SlidersHorizontal}
            eyebrow="profile compare"
            title="Mode & Session Profiles"
            description="draft profile과 active session snapshot을 하단 비교 섹션에 배치해 메인 action rail과 분리합니다."
          />
        </div>
        <ExecutionModesPanel />
      </div>
    </div>
  );
}
