import { useState } from "react";

import { DecisionRail } from "./DecisionRail";
import { CognitionLoopLane } from "./CognitionLoopLane";
import { LoopTimeline } from "./LoopTimeline";
import { RobotViewer } from "./RobotViewer";
import { TopStatusStrip } from "./TopStatusStrip";
import type { LoopStageId } from "./liveLoopStages";

export function LiveLoopWorkspace() {
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [selectedFrameId, setSelectedFrameId] = useState<number | null>(null);
  const [selectedStageId, setSelectedStageId] = useState<LoopStageId>("gateway");

  return (
    <div className="space-y-6">
      <TopStatusStrip />

      <div className="grid grid-cols-1 gap-6 2xl:grid-cols-12">
        <div className="2xl:col-span-5">
          <RobotViewer
            selectedTrackId={selectedTrackId}
            onSelectTrackId={setSelectedTrackId}
            selectedFrameId={selectedFrameId}
            onSelectFrameId={setSelectedFrameId}
          />
        </div>

        <div className="2xl:col-span-4">
          <CognitionLoopLane selectedStageId={selectedStageId} onSelectStageId={setSelectedStageId} />
        </div>

        <div className="2xl:col-span-3">
          <DecisionRail selectedStageId={selectedStageId} />
        </div>
      </div>

      <LoopTimeline />
    </div>
  );
}
