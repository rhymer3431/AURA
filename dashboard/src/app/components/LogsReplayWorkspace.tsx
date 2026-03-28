import { useMemo, useState } from "react";
import { TableProperties } from "lucide-react";

import { useDashboard } from "../state";
import { LogsWidget } from "./SystemStatusWidgets";
import { ConsolePanel, ConsoleSectionTitle } from "./console-ui";

function formatTimestamp(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return new Date(value * 1000).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function LogsReplayWorkspace() {
  const { state } = useDashboard();
  const [taskFilter, setTaskFilter] = useState("");
  const [recoveryFilter, setRecoveryFilter] = useState("");
  const [s2Filter, setS2Filter] = useState("");

  const filteredTrace = useMemo(() => {
    return (state?.cognitionTrace ?? []).filter((item) => {
      const matchesTask = taskFilter.trim() === "" || item.taskId.toLowerCase().includes(taskFilter.trim().toLowerCase());
      const matchesRecovery =
        recoveryFilter.trim() === "" || item.recoveryState.toLowerCase().includes(recoveryFilter.trim().toLowerCase());
      const matchesS2 = s2Filter.trim() === "" || item.s2RawText.toLowerCase().includes(s2Filter.trim().toLowerCase());
      return matchesTask && matchesRecovery && matchesS2;
    });
  }, [recoveryFilter, s2Filter, state?.cognitionTrace, taskFilter]);
  const recentMatches = [...filteredTrace].slice(-5).reverse();
  const latestMatch = recentMatches[0] ?? null;

  return (
    <div className="space-y-5">
      <ConsolePanel className="flex flex-col gap-4">
        <ConsoleSectionTitle
          icon={TableProperties}
          eyebrow="frame replay"
          title="Frame Trace Table"
          description="특정 task, recovery state, S2 raw output 기준으로 프레임 단위 cognition trace를 걸러봅니다."
        />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">matched frames</div>
            <div className="dashboard-mono text-[15px] font-semibold text-[var(--foreground)]">{filteredTrace.length}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">latest decision</div>
            <div className="text-[12px] font-medium text-[var(--foreground)]">{latestMatch?.s2DecisionMode || "no_s2"}</div>
          </div>
          <div className="dashboard-field">
            <div className="dashboard-eyebrow mb-1">latest recovery</div>
            <div className="text-[12px] font-medium text-[var(--foreground)]">{latestMatch?.recoveryState || "clear"}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <input
            className="dashboard-input"
            placeholder="filter task id"
            value={taskFilter}
            onChange={(event) => setTaskFilter(event.target.value)}
          />
          <input
            className="dashboard-input"
            placeholder="filter recovery state"
            value={recoveryFilter}
            onChange={(event) => setRecoveryFilter(event.target.value)}
          />
          <input
            className="dashboard-input"
            placeholder="search S2 raw output"
            value={s2Filter}
            onChange={(event) => setS2Filter(event.target.value)}
          />
        </div>

        <div className="dashboard-scroll overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-y-2 text-left text-[11px]">
            <thead>
              <tr className="text-[var(--text-tertiary)]">
                <th className="px-2 py-1">timestamp</th>
                <th className="px-2 py-1">frame_id</th>
                <th className="px-2 py-1">task_id</th>
                <th className="px-2 py-1">detection_count</th>
                <th className="px-2 py-1">selected_target</th>
                <th className="px-2 py-1">memory_hit_count</th>
                <th className="px-2 py-1">s2_raw_text</th>
                <th className="px-2 py-1">decision_mode</th>
                <th className="px-2 py-1">plan / traj</th>
                <th className="px-2 py-1">command</th>
                <th className="px-2 py-1">recovery</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrace.length === 0 ? (
                <tr>
                  <td colSpan={11} className="rounded-[18px] bg-[var(--surface-2)] px-3 py-4 text-[var(--text-secondary)]">
                    No trace rows match the current filters.
                  </td>
                </tr>
              ) : (
                [...filteredTrace].reverse().map((item) => (
                  <tr key={`${item.frameId}-${item.planVersion}-${item.activeCommandType}`} className="rounded-[18px] bg-[var(--surface-2)] text-[var(--foreground)]">
                    <td className="rounded-l-[18px] px-2 py-2">{formatTimestamp(item.timestamp ?? null)}</td>
                    <td className="px-2 py-2 dashboard-mono">{item.frameId}</td>
                    <td className="px-2 py-2">{item.taskId || "n/a"}</td>
                    <td className="px-2 py-2">{item.detectionCount}</td>
                    <td className="px-2 py-2">{item.selectedTarget || "none"}</td>
                    <td className="px-2 py-2">{item.memoryObjectCount}/{item.memoryPlaceCount}</td>
                    <td className="max-w-[220px] truncate px-2 py-2">{item.s2RawText || "n/a"}</td>
                    <td className="px-2 py-2">{item.s2DecisionMode || "n/a"}</td>
                    <td className="px-2 py-2">v{item.planVersion} / v{item.trajVersion}</td>
                    <td className="px-2 py-2">{item.activeCommandType || item.actionStatus || "idle"}</td>
                    <td className="rounded-r-[18px] px-2 py-2">{item.recoveryState}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </ConsolePanel>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12 xl:items-start">
        <div className="xl:col-span-7">
          <ConsolePanel className="flex flex-col gap-3.5">
            <ConsoleSectionTitle
              icon={TableProperties}
              eyebrow="replay pivots"
              title="Matched Frame Summary"
              description="현재 필터에 걸린 최신 프레임들을 빠르게 훑어보며 replay pivot을 잡습니다."
            />
            <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2">
              {recentMatches.length === 0 ? (
                <div className="dashboard-field md:col-span-2">
                  <div className="dashboard-eyebrow mb-1">status</div>
                  <div className="text-[12px] text-[var(--text-secondary)]">No trace rows match the current filters.</div>
                </div>
              ) : (
                recentMatches.map((item) => (
                  <div key={`${item.frameId}-${item.planVersion}-${item.activeCommandType}-summary`} className="dashboard-field">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-[12px] font-semibold text-[var(--foreground)]">frame {item.frameId}</div>
                      <div className="dashboard-mono text-[10px] text-[var(--text-tertiary)]">{formatTimestamp(item.timestamp ?? null)}</div>
                    </div>
                    <div className="mt-1 text-[11px] text-[var(--text-secondary)]">
                      {item.s2DecisionMode || "no_s2"} · {item.activeCommandType || item.actionStatus || "idle"} · {item.recoveryState}
                    </div>
                    <div className="mt-2 text-[11px] text-[var(--foreground)] break-words">{item.s2RawText || item.actionReason || item.recoveryReason || "no detail"}</div>
                  </div>
                ))
              )}
            </div>
          </ConsolePanel>
        </div>

        <div className="xl:col-span-5">
          <LogsWidget />
        </div>
      </div>
    </div>
  );
}
