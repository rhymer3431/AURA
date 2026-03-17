import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Play,
  Radio,
  Send,
  Square,
  Workflow,
  Wrench,
  Ban,
} from "lucide-react";

import { useDashboard } from "../state";
import {
  asRecord,
  booleanValue,
  dashboardMockModeReason,
  formatMeters,
  formatMs,
  formatSeconds,
  isDashboardMockMode,
  processByName,
  serviceSnapshot,
  statusLabel,
  statusTone,
  stringValue,
} from "../selectors";
import { RobotViewer } from "./RobotViewer";

function toneClasses(status: string) {
  if (statusTone(status) === "green") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (statusTone(status) === "amber") {
    return "border-amber-200 bg-amber-50 text-amber-700";
  }
  if (statusTone(status) === "red") {
    return "border-red-200 bg-red-50 text-red-700";
  }
  return "border-slate-200 bg-slate-50 text-slate-600";
}

function statusDot(status: string) {
  if (statusTone(status) === "green") {
    return "bg-emerald-500";
  }
  if (statusTone(status) === "amber") {
    return "bg-amber-500";
  }
  if (statusTone(status) === "red") {
    return "bg-red-500";
  }
  return "bg-slate-400";
}

function StatusChip({ label, status }: { label: string; status: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium ${toneClasses(status)}`}>
      <span className={`size-1.5 rounded-full ${statusDot(status)}`} />
      {label}: {statusLabel(status)}
    </span>
  );
}

function SummaryCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-2xl bg-white px-4 py-3 shadow-sm">
      <div className="text-[11px] text-black/45 mb-1">{label}</div>
      <div className="text-[16px] font-semibold text-black leading-tight">{value}</div>
      {hint && <div className="text-[11px] text-black/45 mt-1">{hint}</div>}
    </div>
  );
}

function MissionStatusPanel() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const actionStatus = asRecord(runtime.actionStatus);
  const currentPhase = stringValue(runtime.interactivePhase || runtime.plannerControlMode, "idle");
  const routeEnabled = booleanValue(runtime.globalRouteEnabled);
  const routeActive = booleanValue(runtime.globalRouteActive);
  const routeSummary = routeEnabled
    ? `${routeActive ? "active" : "idle"} ${Number(runtime.globalRouteWaypointIndex ?? 0)}/${Number(runtime.globalRouteWaypointCount ?? 0)}`
    : "disabled";
  const lastEvent = state?.session.lastEvent?.message ?? "";
  const plannerMode = state?.session.config?.plannerMode ?? "inactive";

  return (
    <section className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col gap-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
            <Workflow className="size-4" />
            Mission Control
          </div>
          <h3 className="text-[18px] font-semibold text-black">Current Task State</h3>
          <p className="text-[12px] text-black/45 mt-1">
            운영 판단에 필요한 phase, command, route, failure reason만 남겼습니다.
          </p>
        </div>
        <StatusChip label="Phase" status={currentPhase === "" ? "idle" : currentPhase} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <SummaryCard label="Planner Mode" value={plannerMode} />
        <SummaryCard label="Control Mode" value={stringValue(runtime.plannerControlMode, "idle")} />
        <SummaryCard label="Goal Distance" value={formatMeters(runtime.goalDistanceM, "n/a")} />
        <SummaryCard label="Trajectory Freshness" value={formatSeconds(runtime.staleSec, "n/a")} />
        <SummaryCard label="Active Command" value={stringValue(runtime.activeCommandType, "none")} />
        <SummaryCard
          label="Route Progress"
          value={routeSummary}
          hint={
            routeEnabled && Array.isArray(runtime.globalRouteGoalXy)
              ? `goal ${runtime.globalRouteGoalXy[0]}, ${runtime.globalRouteGoalXy[1]}`
              : undefined
          }
        />
      </div>

      <div className="rounded-2xl bg-white px-4 py-4 shadow-sm">
        <div className="flex items-center gap-2 text-[12px] font-medium text-black/70 mb-2">
          <AlertTriangle className="size-4 text-black/35" />
          Immediate Attention
        </div>
        <div className="text-[14px] font-medium text-black break-words">
          {stringValue(actionStatus.reason, "clear")}
        </div>
        <div className="text-[11px] text-black/45 mt-2">
          action state {stringValue(actionStatus.state, "n/a")}
        </div>
      </div>

      <div className="rounded-2xl border border-black/6 bg-black/[0.02] px-4 py-3">
        <div className="text-[11px] text-black/45 mb-1">Latest Notice</div>
        <div className="text-[12px] text-black/75 break-words">
          {lastEvent === "" ? "no recent notice" : lastEvent}
        </div>
      </div>
    </section>
  );
}

function OperationsHealthPanel() {
  const { state } = useDashboard();
  const sensors = asRecord(state?.sensors);
  const transport = asRecord(state?.transport);
  const navdp = serviceSnapshot(state, "navdp");
  const dual = serviceSnapshot(state, "dual");
  const system2 = processByName(state, "system2");
  const requiredFailures = useMemo(
    () =>
      (state?.processes ?? []).filter((process) => process.required && process.state !== "running" && process.state !== "not_required").length,
    [state?.processes],
  );

  return (
    <section className="bg-[#F7F9FB] rounded-3xl p-6 h-full flex flex-col gap-5">
      <div>
        <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
          <Camera className="size-4" />
          Sensor & Viewer Health
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusChip label="RGB" status={booleanValue(sensors.rgbAvailable) ? "ok" : "inactive"} />
          <StatusChip label="Depth" status={booleanValue(sensors.depthAvailable) ? "ok" : "inactive"} />
          <StatusChip label="Pose" status={booleanValue(sensors.poseAvailable) ? "ok" : "inactive"} />
          <StatusChip label="Viewer" status={booleanValue(transport.peerActive) ? "connected" : "inactive"} />
        </div>
        <div className="text-[12px] text-black/55 mt-3">
          frame freshness <span className="font-medium text-black">{formatMs(transport.frameAgeMs, "n/a")}</span>
        </div>
      </div>

      <div className="h-px bg-black/5" />

      <div>
        <div className="flex items-center gap-2 text-[12px] text-black/45 mb-3">
          <Radio className="size-4" />
          Dependency Summary
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <SummaryCard label="NavDP" value={statusLabel(stringValue(navdp.status, "unknown"))} hint={formatMs(navdp.latencyMs, "n/a")} />
          <SummaryCard label="Dual" value={statusLabel(stringValue(dual.status, "inactive"))} hint={formatMs(dual.latencyMs, "n/a")} />
          <SummaryCard label="System2" value={statusLabel(stringValue(system2?.state, "inactive"))} />
          <SummaryCard label="Required Failures" value={String(requiredFailures)} />
        </div>
      </div>
    </section>
  );
}

function OperationsQuickControls() {
  const { form, loading, startSession, stopSession, submitTask, cancelTask, state } = useDashboard();
  const [instruction, setInstruction] = useState("");
  const sessionConfig = state?.session.config;
  const locomotionConfig = form.locomotionConfig;
  const mockMode = isDashboardMockMode(state ?? null);
  const mockModeReason = dashboardMockModeReason(state ?? null);
  const isInteractiveSession = state?.session.active === true && sessionConfig?.plannerMode === "interactive";
  const isPointGoalValid =
    form.plannerMode !== "pointgoal" ||
    (Number.isFinite(Number(form.goalX)) && Number.isFinite(Number(form.goalY)));
  const isLocomotionConfigValid =
    Number.isFinite(Number(locomotionConfig.actionScale)) &&
    Number(locomotionConfig.actionScale) > 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxVx)) &&
    Number(locomotionConfig.cmdMaxVx) >= 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxVy)) &&
    Number(locomotionConfig.cmdMaxVy) >= 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxWz)) &&
    Number(locomotionConfig.cmdMaxWz) > 0;

  return (
    <section className="bg-[#F7F9FB] rounded-3xl p-6">
      <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
        <Wrench className="size-4" />
        Quick Controls
      </div>
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <StatusChip label="Session" status={state?.session.active ? "running" : "inactive"} />
        <StatusChip label="Planner" status={sessionConfig?.plannerMode ?? form.plannerMode} />
        <StatusChip label="Viewer" status={booleanValue(state?.transport.viewerEnabled) ? "ok" : "inactive"} />
      </div>

      {mockMode && (
        <div className="mb-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-[12px] text-amber-800">
          {mockModeReason}
        </div>
      )}

      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => void startSession()}
          disabled={mockMode || loading || !isPointGoalValid || !isLocomotionConfigValid}
          className="inline-flex items-center gap-2 rounded-xl bg-black px-4 py-2 text-[13px] text-white disabled:opacity-50"
        >
          <Play className="size-4" />
          Start Stack
        </button>
        <button
          onClick={() => void stopSession()}
          disabled={mockMode || loading || state?.session.active !== true}
          className="inline-flex items-center gap-2 rounded-xl border border-black/10 bg-white px-4 py-2 text-[13px] disabled:opacity-50"
        >
          <Square className="size-4" />
          Stop Stack
        </button>
        <button
          onClick={() => void cancelTask()}
          disabled={mockMode || !isInteractiveSession}
          className="inline-flex items-center gap-2 rounded-xl border border-black/10 bg-white px-4 py-2 text-[13px] disabled:opacity-40"
        >
          <Ban className="size-4" />
          Cancel Task
        </button>
      </div>

      <div className="flex flex-wrap gap-3">
        <input
          className="min-w-[280px] flex-1 rounded-xl border border-black/10 bg-white px-3 py-2 text-[13px]"
          placeholder={
            mockMode
              ? "mock mode에서는 task를 제출할 수 없습니다"
              : isInteractiveSession
                ? "현재 세션에 자연어 task를 제출합니다"
                : "interactive 세션이 활성화되면 task를 제출할 수 있습니다"
          }
          value={instruction}
          onChange={(event) => setInstruction(event.target.value)}
          disabled={mockMode || !isInteractiveSession}
        />
        <button
          onClick={() => {
            void submitTask(instruction);
            setInstruction("");
          }}
          disabled={mockMode || !isInteractiveSession || instruction.trim() === ""}
          className="inline-flex items-center gap-2 rounded-xl bg-sky-500 px-4 py-2 text-[13px] text-white disabled:opacity-40"
        >
          <Send className="size-4" />
          Submit Task
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-4 mt-4 text-[11px] text-black/45">
        <span>
          draft mode <span className="font-medium text-black">{form.plannerMode} / {form.launchMode}</span>
        </span>
        <span className="truncate">
          locomotion <span className="font-medium text-black">{form.locomotionConfig.actionScale} / {form.locomotionConfig.onnxDevice}</span>
        </span>
        {form.plannerMode === "pointgoal" && (
          <span>
            goal <span className="font-medium text-black">{form.goalX}, {form.goalY}</span>
          </span>
        )}
      </div>
    </section>
  );
}

export function OperationsPage() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-8">
          <RobotViewer />
        </div>
        <div className="xl:col-span-4">
          <MissionStatusPanel />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-7">
          <OperationsQuickControls />
        </div>
        <div className="xl:col-span-5">
          <OperationsHealthPanel />
        </div>
      </div>
    </div>
  );
}
