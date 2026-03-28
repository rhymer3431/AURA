import { DecisionRail } from "./DecisionRail";
import { LoopTimeline } from "./LoopTimeline";

export function NavigationControlPanel() {
  const selectedStageId = "s2" as const;

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      <DecisionRail selectedStageId={selectedStageId} />
      <LoopTimeline />
    </div>
  );
}
