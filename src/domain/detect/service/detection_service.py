from typing import List

from domain.detect.entity.detection import DetectedObject
from domain.detect.repository.detection_model_port import DetectionModelPort


class DetectionService:
    """
    Application-facing domain service that wraps a detection model port.
    Keeps orchestration logic out of infrastructure adapters.
    """

    def __init__(self, model_port: DetectionModelPort):
        self.model_port = model_port

    def track(self, frame) -> List[DetectedObject]:
        return self.model_port.track(frame)
