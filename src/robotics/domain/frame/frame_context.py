# src/robotics/domain/frame/frame_context.py

from dataclasses import dataclass, field
from typing import List, Optional, Any
from robotics.domain.detection.detected_object import DetectedObject
from robotics.domain.tracking.tracked_object import TrackedObject


@dataclass
class FrameContext:
    """
    FrameContext는 프레임 처리 단계들 사이에서 
    데이터를 표준화하여 전달하는 Domain Layer 모델이다.

    Detection → Tracking → SGCL → Policy 등 
    모든 UseCase 단계에서 공유된다.
    """

    frame_id: int
    timestamp: float
    raw_frame: Any                           # np.ndarray (하지만 infra 의존성 제거 위해 Any로 유지)

    detections: List[DetectedObject] = field(default_factory=list)
    tracked_objects: List[TrackedObject] = field(default_factory=list)

    scene_graph: Optional[Any] = None         # SceneGraph Domain 모델 (추가 예정)
    sgcl_result: Optional[Any] = None         # 위험 판단 결과
    trajectory_prediction: Optional[Any] = None
    policy_output: Optional[Any] = None       # LLM/Policy 결정 결과

    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def has_tracked(self) -> bool:
        return len(self.tracked_objects) > 0

    def get_frame_size(self):
        """View 모듈에서 Bounding Box와 함께 사용할 수 있도록 크기 반환."""
        if self.raw_frame is None:
            return None
        return self.raw_frame.shape[:2]  # (H, W)
