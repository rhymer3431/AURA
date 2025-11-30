from dataclasses import dataclass, field
from typing import Any, List, Optional

from domain.detect.entity.detection import DetectedObject


@dataclass
class SceneState:
    """Standard data carrier exchanged across realtime pipeline stages."""

    frame_id: int
    timestamp: float
    raw_frame: Any

    detections: List[DetectedObject] = field(default_factory=list)
    scene_graph: Optional[Any] = None
    sgcl_result: Optional[Any] = None
    trajectory_prediction: Optional[Any] = None
    policy_output: Optional[Any] = None

    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def get_frame_size(self):
        if self.raw_frame is None:
            return None
        return self.raw_frame.shape[:2]
