# src/robotics/interfaces/viz/opencv_overlay.py

import cv2
from robotics.domain.detection.detected_object import DetectedObject


class OpenCVOverlayVisualizer:
    def __init__(self, show_track_id=True, show_score=True):
        self.show_track_id = show_track_id
        self.show_score = show_score

    def draw(self, frame, detections: list[DetectedObject]):
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if self.show_score:
                label += f" {det.score:.2f}"
            if self.show_track_id and det.track_id is not None:
                label = f"ID:{det.track_id} | " + label

            cv2.putText(
                vis, label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        return vis
