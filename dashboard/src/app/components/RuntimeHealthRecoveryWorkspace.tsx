import { ShieldAlert } from "lucide-react";

import { ArtifactsStoragePanel } from "./ArtifactsStoragePanel";
import { ExternalServicesPanel } from "./ExternalServicesPanel";
import {
  IpcOrchestrationWidget,
  MainControlServerWidget,
  ProcessesWidget,
  SensorsWidget,
} from "./SystemStatusWidgets";
import { useDashboard } from "../state";
import { ConsolePanel, ConsoleSectionTitle } from "./console-ui";

export function RuntimeHealthRecoveryWorkspace() {
  const { state } = useDashboard();
  const transitions = [...(state?.recoveryTransitions ?? [])].reverse();

  return (
    <div className="space-y-6">
      <ExternalServicesPanel />

      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-12">
        <div className="grid grid-cols-1 gap-4 2xl:col-span-7 xl:grid-cols-2">
          <ProcessesWidget />
          <SensorsWidget />
          <IpcOrchestrationWidget />
          <MainControlServerWidget />
        </div>

        <div className="space-y-4 2xl:col-span-5">
          <ConsolePanel>
            <ConsoleSectionTitle
              icon={ShieldAlert}
              eyebrow="recovery machine"
              title="Recovery Timeline"
              description="retry, backoff, safe-stop 전이와 최근 recovery reason을 한 카드에 모읍니다."
              className="mb-3"
            />
            <div className="space-y-2">
              {transitions.length === 0 ? (
                <div className="rounded-[16px] border border-[rgba(var(--ink-rgb),0.06)] bg-[var(--surface-2)] px-3.5 py-3 text-[12px] text-[var(--text-secondary)]">
                  No recovery transitions captured yet.
                </div>
              ) : (
                transitions.map((item, index) => (
                  <div key={`${item.to}-${index}`} className="dashboard-field">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-[12px] font-medium text-[var(--foreground)]">
                        {item.from}
                        {" -> "}
                        {item.to}
                      </div>
                      <div className="dashboard-mono text-[10px] text-[var(--text-tertiary)]">retry {item.retryCount}</div>
                    </div>
                    <div className="mt-1 text-[11px] text-[var(--text-secondary)]">{item.reason || "no reason"}</div>
                  </div>
                ))
              )}
            </div>
          </ConsolePanel>

          <ArtifactsStoragePanel />
        </div>
      </div>
    </div>
  );
}
