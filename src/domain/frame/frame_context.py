# src/robotics/domain/frame/frame_context.py

from dataclasses import dataclass, field
from typing import List, Optional, Any
from domain.detection.detected_object import DetectedObject


@dataclass
class FrameContext:
    """
    FrameContext는 프레임 처리 단계 간 교환되는 표준 데이터 컨텍스트이다.
    Detection(Tracking 포함) → SGCL → GRIN → Policy 등 모든 UseCase 경로에서 공통 사용.

    detections 리스트에는 track_id가 포함된 DetectedObject를 저장한다.
    """

    frame_id: int
    timestamp: float
    raw_frame: Any  # np.ndarray 등. Infra 의존성 제거 위해 Any

    # Tracking 포함된 탐지 결과
    detections: List[DetectedObject] = field(default_factory=list)

    # Scene Graph (도메인 모델이 준비되면 타입 지정 예정)
    scene_graph: Optional[Any] = None
    sgcl_result: Optional[Any] = None
    trajectory_prediction: Optional[Any] = None
    policy_output: Optional[Any] = None

    def has_detections(self) -> bool:
        return len(self.detections) > 0

    def get_frame_size(self):
        if self.raw_frame is None:
            return None
        return self.raw_frame.shape[:2]  # (H, W)
