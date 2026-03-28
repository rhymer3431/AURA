import { useState } from "react";

import { CognitionLoopLane } from "./CognitionLoopLane";
import type { LoopStageId } from "./liveLoopStages";

export function PipelineFlow() {
  const [selectedStageId, setSelectedStageId] = useState<LoopStageId>("gateway");
  return <CognitionLoopLane selectedStageId={selectedStageId} onSelectStageId={setSelectedStageId} />;
}
