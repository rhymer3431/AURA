# src/robotics/domain/tracking/detection_port.py

from abc import ABC, abstractmethod
from typing import List, Protocol
from robotics.domain.detection.detected_object import DetectedObject


class DetectionPort(ABC):
    """
    Domain Layer에서 사용하는 Detector Port Interface.

    모든 Detector 어댑터는 이 인터페이스를 구현해야 함.
    예: YOLOWorldAdapter, FasterRCNNAdapter, GroundingDINOAdapter 등.
    """

    @abstractmethod
    def detect(self, frame) -> List[DetectedObject]:
        """
        입력: BGR 이미지 (np.ndarray)
        출력: DetectedObject 리스트
        """
        raise NotImplementedError
