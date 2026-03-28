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
    <div className="space-y-5">
      <div
        className="sticky top-0 z-10 rounded-[20px] pb-1 backdrop-blur"
        style={{ background: "color-mix(in srgb, var(--background) 78%, transparent)" }}
      >
        <TopStatusStrip />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12 xl:items-start">
        <div className="xl:col-span-5">
          <RobotViewer
            selectedTrackId={selectedTrackId}
            onSelectTrackId={setSelectedTrackId}
            selectedFrameId={selectedFrameId}
            onSelectFrameId={setSelectedFrameId}
          />
        </div>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:col-span-7 xl:grid-cols-7">
          <div className="xl:col-span-4">
            <CognitionLoopLane selectedStageId={selectedStageId} onSelectStageId={setSelectedStageId} />
          </div>

          <div className="xl:col-span-3">
            <DecisionRail selectedStageId={selectedStageId} />
          </div>
        </div>
      </div>

      <LoopTimeline />
    </div>
  );
}
