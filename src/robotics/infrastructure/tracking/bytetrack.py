# src/robotics/infrastructure/tracking/bytetrack_ultra.py

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from robotics.domain.ports.detection_port import DetectionResult
from robotics.domain.ports.tracking_port import TrackingPort


class ByteTrackAdapter(TrackingPort):
    """
    Ultralytics BYTETracker adapter to Track Port interface.
    DetectionResult(track_id=None) -> DetectionResult(track_id=assigned)
    """

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=30,
        )

    def update_tracks(
        self,
        detections: List[DetectionResult],
        frame_shape: Tuple[int, int, int],
    ) -> List[DetectionResult]:

        if not detections:
            self.tracker.update([], frame_shape[:2])
            return []

        # 1) Convert to ByteTrack input format [x1,y1,x2,y2,score,class]
        det_array = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det_array.append([x1, y1, x2, y2, det.score, det.class_id])

        det_array = np.array(det_array, dtype=float)

        # 2) Update tracker
        online_targets = self.tracker.update(det_array, frame_shape[:2])

        tracked_results: List[DetectionResult] = []

        # 3) Merge tracking result back
        for t in online_targets:
            track_id = int(t.track_id)
            x1, y1, x2, y2 = map(int, t.tlbr)  # top-left bottom-right

            # find closest detection for class + score
            cls_id = int(t.class_id) if hasattr(t, "class_id") else -1
            class_name = "unknown"

            # domain에서 YOLO의 모델명을 모르므로 class_name은 unknown 처리
            # 필요 시 application에서 다시 매핑 가능

            tracked_results.append(
                DetectionResult(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    score=float(t.score),
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

        return tracked_results
