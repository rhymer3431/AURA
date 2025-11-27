# src/robotics/infrastructure/detection/yolo_world_ultra.py

from __future__ import annotations
import torch
from typing import List
from ultralytics import YOLOWorld

from robotics.domain.ports.detection_port import DetectionPort, DetectionResult


class YoloWorldDetector(DetectionPort):
    """
    Ultralytics YOLO-World Detector Adapter.
    Domain layer에서는 YOLO 구현체를 알지 못하고,
    오직 DetectionPort 인터페이스로만 접근합니다.
    """

    def __init__(
        self,
        weight_path: str,
        device: str = "auto",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        classes: List[str] | None = None,
    ):
        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLOWorld(weight_path)
        self.model.to(device)
        self.device = device

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = set(classes) if classes else None

    def detect(self, frame_bgr) -> List[DetectionResult]:
        """
        입력: BGR 프레임 (OpenCV)
        출력: 추상화된 DetectionResult 리스트
        """

        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        boxes = results.boxes
        detections: List[DetectionResult] = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            class_name = self.model.names[cls_id]

            # 클래스 필터링
            if self.class_filter and class_name not in self.class_filter:
                continue

            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            detection = DetectionResult(
                track_id=None,  # TrackingPort에서 채울 것
                bbox=(x1, y1, x2, y2),
                score=score,
                class_id=cls_id,
                class_name=class_name,
            )
            detections.append(detection)

        return detections
