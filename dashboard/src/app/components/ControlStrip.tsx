import { useState } from "react";
import { Play, Square, Send, Ban } from "lucide-react";

import { useDashboard } from "../state";
import { ConsoleBadge, ConsolePanel, ConsoleSectionTitle } from "./console-ui";

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
    <label className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="accent-[var(--signal-cyan)]"
      />
      {label}
    </label>
  );
}

export function ControlStrip() {
  const { bootstrap, form, setForm, startSession, stopSession, submitTask, cancelTask, state, loading } = useDashboard();
  const [instruction, setInstruction] = useState("");
  const sessionConfig = state?.session.config;
  const locomotionConfig = form.locomotionConfig;
  const runtimeMode = String((state?.runtime.executionMode ?? state?.runtime.modes?.executionMode ?? "IDLE"));
  const canSubmitTask = state?.session.active === true;
  const isLocomotionConfigValid =
    Number.isFinite(Number(locomotionConfig.actionScale)) &&
    Number(locomotionConfig.actionScale) > 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxVx)) &&
    Number(locomotionConfig.cmdMaxVx) >= 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxVy)) &&
    Number(locomotionConfig.cmdMaxVy) >= 0 &&
    Number.isFinite(Number(locomotionConfig.cmdMaxWz)) &&
    Number(locomotionConfig.cmdMaxWz) > 0;
  const scenePresets = bootstrap?.scenePresets ?? ["warehouse", "interioragent", "interior agent kujiale 3"];
  const lastEvent = state?.session.lastEvent?.message ?? "";

  return (
    <ConsolePanel className="p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <ConsoleSectionTitle
          icon={Play}
          eyebrow="session control"
          title="Runtime Control Strip"
          description="launch profile, locomotion constraints, and task submission live on one rail"
        />
        <ConsoleBadge tone={state?.session.active ? "emerald" : "amber"}>
          {state?.session.active ? "stack running" : "stack stopped"}
        </ConsoleBadge>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <div className="dashboard-eyebrow mb-1">launch mode</div>
            <select
              className="dashboard-input"
              value={form.launchMode}
              onChange={(event) => setForm({ launchMode: event.target.value as "gui" | "headless" })}
            >
              <option value="gui">gui</option>
              <option value="headless">headless</option>
            </select>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">scene preset</div>
            <select
              className="dashboard-input"
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
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => void startSession()}
              disabled={loading || !isLocomotionConfigValid}
              className="dashboard-button-primary disabled:opacity-50"
            >
              <Play className="size-4" />
              Start Stack
            </button>
            <button
              onClick={() => void stopSession()}
              disabled={loading || state?.session.active !== true}
              className="dashboard-button-secondary disabled:opacity-50"
            >
              <Square className="size-4" />
              Stop Stack
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <Toggle label="viewer publish" checked={form.viewerEnabled} onChange={(checked) => setForm({ viewerEnabled: checked })} />
          <Toggle label="memory store" checked={form.memoryStore} onChange={(checked) => setForm({ memoryStore: checked })} />
          <Toggle label="detection" checked={form.detectionEnabled} onChange={(checked) => setForm({ detectionEnabled: checked })} />
          <div className="ml-auto dashboard-micro">
            active session: <span className="text-[var(--foreground)]">{state?.session.active ? "running" : "stopped"}</span>
          </div>
        </div>

        {!isLocomotionConfigValid && (
          <div className="rounded-xl border border-[var(--tone-amber-border)] bg-[var(--tone-amber-bg)] px-3 py-2 text-[12px] text-[var(--tone-amber-fg)]">
            {"locomotion config는 `action scale > 0`, `cmd max vx/vy >= 0`, `cmd max wz > 0` 이어야 합니다."}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
          <div>
            <div className="dashboard-eyebrow mb-1">action scale</div>
            <input
              className="dashboard-input"
              value={locomotionConfig.actionScale}
              onChange={(event) =>
                setForm({ locomotionConfig: { ...locomotionConfig, actionScale: event.target.value } })
              }
            />
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">onnx device</div>
            <select
              className="dashboard-input"
              value={locomotionConfig.onnxDevice}
              onChange={(event) =>
                setForm({
                  locomotionConfig: {
                    ...locomotionConfig,
                    onnxDevice: event.target.value as "auto" | "cuda" | "cpu",
                  },
                })
              }
            >
              <option value="auto">auto</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">cmd max vx</div>
            <input
              className="dashboard-input"
              value={locomotionConfig.cmdMaxVx}
              onChange={(event) =>
                setForm({ locomotionConfig: { ...locomotionConfig, cmdMaxVx: event.target.value } })
              }
            />
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">cmd max vy</div>
            <input
              className="dashboard-input"
              value={locomotionConfig.cmdMaxVy}
              onChange={(event) =>
                setForm({ locomotionConfig: { ...locomotionConfig, cmdMaxVy: event.target.value } })
              }
            />
          </div>
          <div>
            <div className="dashboard-eyebrow mb-1">cmd max wz</div>
            <input
              className="dashboard-input"
              value={locomotionConfig.cmdMaxWz}
              onChange={(event) =>
                setForm({ locomotionConfig: { ...locomotionConfig, cmdMaxWz: event.target.value } })
              }
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <input
            className="dashboard-input flex-1 min-w-[280px]"
            placeholder={canSubmitTask ? "instruction을 입력하면 서버가 실행 모드를 분류합니다" : "running session에서만 task를 제출할 수 있습니다"}
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            disabled={!canSubmitTask}
          />
          <button
            onClick={() => {
              void submitTask(instruction);
              setInstruction("");
            }}
            disabled={!canSubmitTask || instruction.trim() === ""}
            className="dashboard-button-accent disabled:opacity-40"
          >
            <Send className="size-4" />
            Submit Task
          </button>
          <button
            onClick={() => void cancelTask()}
            disabled={!canSubmitTask}
            className="dashboard-button-secondary disabled:opacity-40"
          >
            <Ban className="size-4" />
            Set Idle
          </button>
        </div>

        <div className="dashboard-inset flex flex-wrap items-center gap-4 px-4 py-3 text-[11px]">
          <span>
            runtime:{" "}
            <span className="dashboard-mono text-[var(--foreground)]">
              {runtimeMode} / {sessionConfig?.launchMode ?? form.launchMode}
            </span>
          </span>
          <span className="truncate">
            locomotion:{" "}
            <span className="dashboard-mono text-[var(--foreground)]">
              action {sessionConfig?.locomotionConfig.actionScale ?? Number(locomotionConfig.actionScale)} /{" "}
              {sessionConfig?.locomotionConfig.onnxDevice ?? locomotionConfig.onnxDevice}
            </span>
          </span>
          <span>
            viewer:{" "}
            <span className="dashboard-mono text-[var(--foreground)]">
              {Boolean(state?.transport.viewerEnabled) ? "published" : "inactive"}
            </span>
          </span>
          {lastEvent !== "" && (
            <span className="truncate">
              last event: <span className="dashboard-mono text-[var(--foreground)]">{lastEvent}</span>
            </span>
          )}
        </div>
      </div>
    </ConsolePanel>
  );
}
