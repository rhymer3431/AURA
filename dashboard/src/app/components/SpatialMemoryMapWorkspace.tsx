import { Brain, MapPinned, SearchCheck } from "lucide-react";

import { OccupancyMapPanel } from "./OccupancyMapPanel";
import { useDashboard } from "../state";
import { asRecord, stringValue } from "../selectors";
import { ConsolePanel, ConsoleSectionTitle } from "./console-ui";

export function SpatialMemoryMapWorkspace() {
  const { state } = useDashboard();
  const memory = asRecord(state?.memory);
  const scratchpad = asRecord(memory.scratchpad);
  const selectedTarget = state?.selectedTargetSummary;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12 xl:items-start">
      <div className="xl:col-span-8">
        <OccupancyMapPanel />
      </div>

      <div className="space-y-4 self-start xl:col-span-4 xl:sticky xl:top-0">
        <ConsolePanel className="space-y-5">
          <ConsoleSectionTitle
            icon={MapPinned}
            eyebrow="spatial context"
            title="Target & Memory Context"
            description="map 위 target anchor와 current memory footprint를 같은 시선 흐름으로 정리합니다."
          />
          <section className="space-y-2.5 text-[11px]">
            <div>
              <div className="dashboard-title text-[12px]">World Pose Focus</div>
              <p className="dashboard-subtitle mt-1">현재 타겟 object의 normalized world pose와 navigation anchor를 요약합니다.</p>
            </div>
            <div className="space-y-2">
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">class / track</div>
                <div className="text-[var(--foreground)]">{selectedTarget ? `${selectedTarget.className || "unknown"} / ${selectedTarget.trackId || "n/a"}` : "no selected target"}</div>
              </div>
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">world pose</div>
                <div className="dashboard-mono text-[var(--foreground)]">
                  {selectedTarget?.worldPose ? selectedTarget.worldPose.join(", ") : "n/a"}
                </div>
              </div>
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">nav goal pixel</div>
                <div className="dashboard-mono text-[var(--foreground)]">
                  {selectedTarget?.navGoalPixel ? selectedTarget.navGoalPixel.join(", ") : "n/a"}
                </div>
              </div>
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">source / confidence</div>
                <div className="text-[var(--foreground)]">
                  {selectedTarget ? `${selectedTarget.source} / ${selectedTarget.confidence?.toFixed(2) ?? "n/a"}` : "n/a"}
                </div>
              </div>
            </div>
          </section>

          <section className="space-y-2.5 text-[11px]">
            <div className="flex items-center gap-2">
              <Brain className="size-4 text-[var(--text-tertiary)]" />
              <div>
                <div className="dashboard-title text-[12px]">Remembered Objects & Places</div>
                <p className="dashboard-subtitle mt-1">current memory footprint와 scratchpad task 상태를 공간 해석과 함께 봅니다.</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">objects</div>
                <div className="dashboard-mono text-[var(--foreground)]">{String(memory.objectCount ?? 0)}</div>
              </div>
              <div className="dashboard-field">
                <div className="dashboard-eyebrow mb-1">places</div>
                <div className="dashboard-mono text-[var(--foreground)]">{String(memory.placeCount ?? 0)}</div>
              </div>
              <div className="dashboard-field col-span-2">
                <div className="dashboard-eyebrow mb-1">scratchpad task</div>
                <div className="text-[var(--foreground)]">{stringValue(scratchpad.taskState, "idle")}</div>
              </div>
              <div className="dashboard-field col-span-2">
                <div className="dashboard-eyebrow mb-1">instruction</div>
                <div className="text-[var(--foreground)] break-words">{stringValue(scratchpad.instruction, "idle")}</div>
              </div>
            </div>
          </section>
        </ConsolePanel>

        <ConsolePanel>
          <ConsoleSectionTitle
            icon={SearchCheck}
            eyebrow="semantic retrieval"
            title="Retrieval Candidates"
            description="현재 payload에 retrieval 후보가 없으면 empty state를 유지하고, 후속 instrumentation을 기다립니다."
            className="mb-3"
          />
          <div className="rounded-[18px] border border-dashed border-[rgba(var(--ink-rgb),0.1)] bg-[var(--surface-2)] px-4 py-5 text-[12px] text-[var(--text-secondary)]">
            No retrieval candidate payload yet. This panel is reserved for semantic retrieval confidence, source, and last-seen rows when backend instrumentation lands.
          </div>
        </ConsolePanel>
      </div>
    </div>
  );
}
