# Abstract port for detection/tracking models.
from abc import ABC, abstractmethod
from typing import List

from domain.detect.entity.detection import DetectedObject


class DetectionModelPort(ABC):
    @abstractmethod
    def track(self, frame) -> List[DetectedObject]:
        """Run detection + tracking on a frame and return DetectedObject items."""
        raise NotImplementedError
