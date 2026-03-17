import { AlertTriangle, Route, ScanSearch } from "lucide-react";

import { asRecord, stringValue } from "../selectors";
import { useDashboard } from "../state";
import { ArtifactsStoragePanel } from "./ArtifactsStoragePanel";
import { ExternalServicesPanel } from "./ExternalServicesPanel";
import {
  IpcOrchestrationWidget,
  LogsWidget,
  ProcessesWidget,
} from "./SystemStatusWidgets";

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 text-[12px]">
      <span className="text-black/45">{label}</span>
      <span className="max-w-[65%] break-words text-right font-medium text-black/80">{value}</span>
    </div>
  );
}

function RuntimeDebugPanel() {
  const { state } = useDashboard();
  const runtime = asRecord(state?.runtime);
  const perception = asRecord(state?.perception);
  const capability = asRecord(perception.detectorCapability);
  const sensors = asRecord(state?.sensors);
  const sensorMeta = asRecord(sensors.sensorMeta);
  const captureReport = asRecord(sensors.captureReport);
  const lastStatus = asRecord(runtime.lastStatusEvent);

  return (
    <div className="bg-[#F7F9FB] rounded-3xl p-6 flex flex-col gap-5">
      <div>
        <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
          <ScanSearch className="size-4" />
          Detector & Sensor Debug
        </div>
        <div className="space-y-3 rounded-2xl bg-white p-4 shadow-sm">
          <DetailRow label="Detector Backend" value={stringValue(perception.detectorBackend, "unknown")} />
          <DetailRow label="Selected Reason" value={stringValue(perception.detectorSelectedReason, "n/a")} />
          <DetailRow label="Capability Status" value={stringValue(capability.status, "unknown")} />
          <DetailRow
            label="Capture Note"
            value={stringValue(captureReport.note, stringValue(sensorMeta.fallback_reason, "n/a"))}
          />
          <DetailRow label="Capture Status" value={stringValue(captureReport.status, "n/a")} />
        </div>
      </div>

      <div>
        <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
          <Route className="size-4" />
          Route Debug
        </div>
        <div className="space-y-3 rounded-2xl bg-white p-4 shadow-sm">
          <DetailRow label="Replan Reason" value={stringValue(runtime.globalRouteLastReplanReason, "none")} />
          <DetailRow label="Route Error" value={stringValue(runtime.globalRouteLastError, "clear")} />
          <DetailRow label="Waypoint Progress" value={`${Number(runtime.globalRouteWaypointIndex ?? 0)}/${Number(runtime.globalRouteWaypointCount ?? 0)}`} />
        </div>
      </div>

      <div>
        <div className="flex items-center gap-2 text-[12px] text-black/45 mb-1">
          <AlertTriangle className="size-4" />
          Last Runtime Status
        </div>
        <div className="rounded-2xl bg-white p-4 shadow-sm">
          <DetailRow label="State" value={stringValue(lastStatus.state, "n/a")} />
          <DetailRow label="Reason" value={stringValue(lastStatus.reason, "n/a")} />
        </div>
      </div>
    </div>
  );
}

function DiagnosticsSidebar() {
  return (
    <div className="space-y-6">
      <IpcOrchestrationWidget />
      <RuntimeDebugPanel />
    </div>
  );
}

export function DiagnosticsPage() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-8">
          <ExternalServicesPanel />
        </div>
        <div className="xl:col-span-4">
          <DiagnosticsSidebar />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-4">
          <ProcessesWidget />
        </div>
        <div className="xl:col-span-8">
          <LogsWidget />
        </div>
      </div>

      <ArtifactsStoragePanel />
    </div>
  );
}
