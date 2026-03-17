import { Crosshair, Eye, Layers3, ScanLine, SlidersHorizontal } from "lucide-react";

import { useDashboard } from "../state";
import { asRecord, booleanValue, runtimeComponentLabel, stringValue } from "../selectors";

function ModeBadge({
  label,
  enabled,
}: {
  label: string;
  enabled: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-medium ${
        enabled
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-slate-200 bg-slate-50 text-slate-600"
      }`}
    >
      <span className={`size-1.5 rounded-full ${enabled ? "bg-emerald-500" : "bg-slate-400"}`} />
      {label}
    </span>
  );
}

export function ExecutionModesPanel() {
  const { bootstrap, form, state } = useDashboard();
  const liveConfig = state?.session.config;
  const runtime = asRecord(state?.runtime);
  const plannerModes = bootstrap?.plannerModes ?? ["interactive", "pointgoal"];
  const launchModes = bootstrap?.launchModes ?? ["gui", "headless"];
  const runtimeOwner = stringValue(runtime.ownerDisplayName, "NavigationRuntime");
  const runtimeComponent = runtimeComponentLabel(runtime.ownerComponent, "navigation_runtime");
  const goalText =
    form.plannerMode === "pointgoal"
      ? `${form.goalX || "?"}, ${form.goalY || "?"}`
      : "not used";
  const locomotionText = [
    `action ${form.locomotionConfig.actionScale}`,
    form.locomotionConfig.onnxDevice,
    `vx ${form.locomotionConfig.cmdMaxVx}`,
    `vy ${form.locomotionConfig.cmdMaxVy}`,
    `wz ${form.locomotionConfig.cmdMaxWz}`,
  ].join(" / ");
  const liveGoalText =
    liveConfig?.goal === undefined
      ? "not used"
      : `${liveConfig.goal.x.toFixed(2)}, ${liveConfig.goal.y.toFixed(2)}`;
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
    <div className="bg-[#F7F9FB] rounded-3xl p-6 flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <SlidersHorizontal className="size-4 text-black/40" />
        <div>
          <h3 className="text-[15px] font-semibold text-black">Runtime Ownership & Modes</h3>
          <p className="text-[12px] text-black/50 mt-0.5">NavigationRuntime owner와 draft launch options, 현재 세션 설정을 분리해서 보여줍니다.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="text-[12px] font-semibold text-black/75 mb-4">Draft Session</div>
          <div className="grid grid-cols-2 gap-3 text-[12px]">
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Planner Mode</div>
              <div className="font-medium text-black">{form.plannerMode}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Launch Mode</div>
              <div className="font-medium text-black">{form.launchMode}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Scene Preset</div>
              <div className="font-medium text-black">{form.scenePreset}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Point Goal</div>
              <div className="font-medium text-black">{goalText}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3 col-span-2">
              <div className="text-black/45 mb-1">G1 Locomotion Config</div>
              <div className="font-medium text-black break-all">{locomotionText}</div>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <ModeBadge label="viewer publish" enabled={form.viewerEnabled} />
            <ModeBadge label="memory store" enabled={form.memoryStore} />
            <ModeBadge label="detection" enabled={form.detectionEnabled} />
          </div>
        </div>

        <div className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="text-[12px] font-semibold text-black/75">Active Session</div>
            <span
              className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${
                state?.session.active
                  ? "bg-emerald-50 text-emerald-700"
                  : "bg-amber-50 text-amber-700"
              }`}
            >
              {state?.session.active ? "running" : "idle"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 text-[12px]">
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Planner Mode</div>
              <div className="font-medium text-black">{liveConfig?.plannerMode ?? "inactive"}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Launch Mode</div>
              <div className="font-medium text-black">{liveConfig?.launchMode ?? "inactive"}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Scene Preset</div>
              <div className="font-medium text-black">{liveConfig?.scenePreset ?? "inactive"}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3">
              <div className="text-black/45 mb-1">Point Goal</div>
              <div className="font-medium text-black">{liveGoalText}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3 col-span-2">
              <div className="text-black/45 mb-1">G1 Locomotion Config</div>
              <div className="font-medium text-black break-all">{liveLocomotionText}</div>
            </div>
            <div className="bg-[#F7F9FB] rounded-xl px-3 py-3 col-span-2">
              <div className="text-black/45 mb-1">Runtime Owner</div>
              <div className="font-medium text-black break-all">{runtimeOwner} · {runtimeComponent}</div>
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
        <div className="bg-white rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-medium text-black/75 mb-2">
            <Layers3 className="size-4 text-black/35" />
            Planner Modes
          </div>
          <div className="text-[12px] text-black/60 leading-6">{plannerModes.join(" / ")}</div>
        </div>
        <div className="bg-white rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-medium text-black/75 mb-2">
            <Eye className="size-4 text-black/35" />
            Launch Modes
          </div>
          <div className="text-[12px] text-black/60 leading-6">{launchModes.join(" / ")}</div>
        </div>
        <div className="bg-white rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-medium text-black/75 mb-2">
            <Crosshair className="size-4 text-black/35" />
            Current Control
          </div>
          <div className="text-[12px] text-black/60 leading-6">
            {state?.session.active ? liveConfig?.plannerMode ?? "inactive" : "no active session"}
          </div>
        </div>
        <div className="bg-white rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-2 text-[12px] font-medium text-black/75 mb-2">
            <ScanLine className="size-4 text-black/35" />
            Detection Gate
          </div>
          <div className="text-[12px] text-black/60 leading-6">
            {booleanValue(liveConfig?.detectionEnabled, form.detectionEnabled) ? "enabled" : "disabled"}
          </div>
        </div>
      </div>
    </div>
  );
}
