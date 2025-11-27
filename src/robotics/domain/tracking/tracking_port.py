# src/robotics/domain/tracking/tracking_port.py

from abc import ABC, abstractmethod
from typing import List
from robotics.domain.detection.detected_object import DetectedObject
from robotics.domain.tracking.tracked_object import TrackedObject


class TrackingPort(ABC):
    """
    Domain Layer의 TrackingPort.
    객체 Tracking 기능의 인터페이스(명세)만 정의하며,
    실제 구현(ByteTrack, DeepSORT, OC-SORT 등)은 Infra Adapter에서 담당한다.

    입력: List[DetectedObject]
    출력: List[TrackedObject]

    역할:
    - Detection 단계의 출력을 Tracking 단계의 표준 데이터 구조로 변환
    - 도메인 로직(예: Scene Graph, SGCL, GRIN)이 Tracker 종류와 무관하게 동작하도록 보장
    """

    @abstractmethod
    def track(self, detections: List[DetectedObject]) -> List[TrackedObject]:
        """
        Tracking을 수행하고 동일 프레임에서의 TrackedObject 리스트를 반환한다.

        parameters:
            detections: DetectionPort에서 전달된 DetectedObject 리스트

        returns:
            tracked_objects: Tracking ID, bbox, label, score, velocity 등을 포함하는
                             TrackedObject 리스트
        """
        raise NotImplementedError
