import { useState } from "react";
import { Play, Square, Send, Ban } from "lucide-react";

import { useDashboard } from "../state";

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-[12px] text-black/70">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      {label}
    </label>
  );
}

export function ControlStrip() {
  const { bootstrap, form, setForm, startSession, stopSession, submitTask, cancelTask, state, loading } = useDashboard();
  const [instruction, setInstruction] = useState("");
  const sessionConfig = state?.session.config;
  const isInteractiveSession = state?.session.active === true && sessionConfig?.plannerMode === "interactive";
  const isPointGoalValid =
    form.plannerMode !== "pointgoal" ||
    (Number.isFinite(Number(form.goalX)) && Number.isFinite(Number(form.goalY)));
  const scenePresets = bootstrap?.scenePresets ?? ["warehouse", "interioragent", "interior agent kujiale 3"];
  const lastEvent = state?.session.lastEvent?.message ?? "";

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-5 border border-black/5">
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <div className="text-[11px] text-black/40 mb-1">planner mode</div>
            <select
              className="bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10"
              value={form.plannerMode}
              onChange={(event) => setForm({ plannerMode: event.target.value as "interactive" | "pointgoal" })}
            >
              <option value="interactive">interactive</option>
              <option value="pointgoal">pointgoal</option>
            </select>
          </div>
          <div>
            <div className="text-[11px] text-black/40 mb-1">launch mode</div>
            <select
              className="bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10"
              value={form.launchMode}
              onChange={(event) => setForm({ launchMode: event.target.value as "gui" | "headless" })}
            >
              <option value="gui">gui</option>
              <option value="headless">headless</option>
            </select>
          </div>
          <div>
            <div className="text-[11px] text-black/40 mb-1">scene preset</div>
            <select
              className="bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10"
              value={form.scenePreset}
              onChange={(event) => setForm({ scenePreset: event.target.value })}
            >
              {scenePresets.map((preset) => (
                <option key={preset} value={preset}>
                  {preset}
                </option>
              ))}
            </select>
          </div>
          {form.plannerMode === "pointgoal" && (
            <>
              <div>
                <div className="text-[11px] text-black/40 mb-1">goal x</div>
                <input
                  className="bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10 w-[100px]"
                  value={form.goalX}
                  onChange={(event) => setForm({ goalX: event.target.value })}
                />
              </div>
              <div>
                <div className="text-[11px] text-black/40 mb-1">goal y</div>
                <input
                  className="bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10 w-[100px]"
                  value={form.goalY}
                  onChange={(event) => setForm({ goalY: event.target.value })}
                />
              </div>
            </>
          )}
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => void startSession()}
              disabled={loading || !isPointGoalValid}
              className="inline-flex items-center gap-2 rounded-xl bg-black text-white px-4 py-2 text-[13px] disabled:opacity-50"
            >
              <Play className="size-4" />
              Start Stack
            </button>
            <button
              onClick={() => void stopSession()}
              disabled={loading || state?.session.active !== true}
              className="inline-flex items-center gap-2 rounded-xl bg-white border border-black/10 px-4 py-2 text-[13px] disabled:opacity-50"
            >
              <Square className="size-4" />
              Stop Stack
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <Toggle label="viewer publish" checked={form.viewerEnabled} onChange={(checked) => setForm({ viewerEnabled: checked })} />
          <Toggle label="show depth" checked={form.showDepth} onChange={(checked) => setForm({ showDepth: checked })} />
          <Toggle label="memory store" checked={form.memoryStore} onChange={(checked) => setForm({ memoryStore: checked })} />
          <Toggle label="detection" checked={form.detectionEnabled} onChange={(checked) => setForm({ detectionEnabled: checked })} />
          <div className="ml-auto text-[11px] text-black/40">
            active session: <span className="text-black/80 font-medium">{state?.session.active ? "running" : "stopped"}</span>
          </div>
        </div>

        {!isPointGoalValid && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[12px] text-amber-700">
            pointgoal 모드에서는 numeric `goal x / goal y`가 필요합니다.
          </div>
        )}

        <div className="flex flex-wrap gap-3">
          <input
            className="flex-1 min-w-[280px] bg-white rounded-xl px-3 py-2 text-[13px] border border-black/10"
            placeholder={isInteractiveSession ? "자연어 task를 입력하세요" : "running interactive 세션에서만 task를 제출할 수 있습니다"}
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            disabled={!isInteractiveSession}
          />
          <button
            onClick={() => {
              void submitTask(instruction);
              setInstruction("");
            }}
            disabled={!isInteractiveSession || instruction.trim() === ""}
            className="inline-flex items-center gap-2 rounded-xl bg-sky-500 text-white px-4 py-2 text-[13px] disabled:opacity-40"
          >
            <Send className="size-4" />
            Submit Task
          </button>
          <button
            onClick={() => void cancelTask()}
            disabled={!isInteractiveSession}
            className="inline-flex items-center gap-2 rounded-xl bg-white border border-black/10 px-4 py-2 text-[13px] disabled:opacity-40"
          >
            <Ban className="size-4" />
            Cancel Task
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-[11px] text-black/45">
          <span>
            runtime:{" "}
            <span className="text-black/80 font-medium">
              {sessionConfig?.plannerMode ?? form.plannerMode} / {sessionConfig?.launchMode ?? form.launchMode}
            </span>
          </span>
          <span>
            viewer:{" "}
            <span className="text-black/80 font-medium">
              {Boolean(state?.transport.viewerEnabled) ? "published" : "inactive"}
            </span>
          </span>
          {lastEvent !== "" && (
            <span className="truncate">
              last event: <span className="text-black/75">{lastEvent}</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
