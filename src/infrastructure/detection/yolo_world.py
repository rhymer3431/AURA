# src/robotics/infrastructure/detection/yolo_world_ultra.py

from __future__ import annotations
import torch
from typing import List
from ultralytics import YOLOWorld

from domain.detection.detection_port import DetectionPort
from domain.detection.detected_object import DetectedObject


class YoloWorldAdapter(DetectionPort):
    """
    Ultralytics YOLO-World Detector Adapter.
    ROI:
      - detect() 제거
      - track()에 Tracker 통합
      - Domain Layer에 track_id가 포함된 DetectedObject 전달
    """

    def __init__(
        self,
        weight_path: str,
        device: str = "auto",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        classes: List[str] | None = None,
    ):
        # Device 자동 설정
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLOWorld(weight_path)
        self.model.to(device)
        self.device = device

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = set(classes) if classes else None

    def track(self, raw_frame) -> List[DetectedObject]:
        """
        입력: BGR frame (OpenCV)
        출력: Domain layer DetectedObject with tracking id
        """
        results = self.model.track(
            source=raw_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        if len(results) == 0:
            return []

        boxes = results[0].boxes
        detections: List[DetectedObject] = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            class_name = self.model.names[cls_id]

            # 클래스 필터 조건
            if self.class_filter and class_name not in self.class_filter:
                continue

            # Tracking ID 반환
            # ByteTrack이 id를 못줬을 경우 None으로 처리
            track_id = (
                int(boxes.id[i].item()) if boxes.id is not None else None
            )

            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            detection = DetectedObject(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                score=conf,
                class_id=cls_id,
                class_name=class_name,
            )
            detections.append(detection)

        return detections
