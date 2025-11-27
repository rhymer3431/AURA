# robotics/domain/detection/detection_port.py

from abc import ABC, abstractmethod
from typing import List
from server.src.domain.detection.detected_object import DetectedObject

class DetectionPort(ABC):
    
    @abstractmethod
    def track(self, frame) -> List[DetectedObject]:
        """
        YOLO tracking + detection combined
        Returns detections with tracking IDs
        """
        pass
