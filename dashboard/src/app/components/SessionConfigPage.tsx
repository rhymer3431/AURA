import { ControlStrip } from "./ControlStrip";
import { ExecutionModesPanel } from "./ExecutionModesPanel";

export function SessionConfigPage() {
  return (
    <div className="space-y-6">
      <ControlStrip />
      <ExecutionModesPanel />
    </div>
  );
}
