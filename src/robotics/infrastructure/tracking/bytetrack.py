# src/robotics/infrastructure/tracking/bytetrack.py
from types import SimpleNamespace
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from robotics.domain.tracking.tracked_object import TrackedObject
from robotics.domain.detection.detected_object import DetectedObject


class ByteTrackAdapter:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        aspect_ratio_thresh: float = 1.6,
        min_box_area: float = 10,
        mot20: bool = False,
        frame_rate: int = 30,
    ) -> None:
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            aspect_ratio_thresh=aspect_ratio_thresh,
            min_box_area=min_box_area,
            mot20=mot20,
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def track(self, frame, detections):
        img_h, img_w = frame.shape[:2]
        img_size = (img_h, img_w)

        # 1. Detection 변환
        if len(detections) > 0:
            dets = np.array([
                [
                    *det.bbox,         # x1,y1,x2,y2
                    det.score,
                    det.class_id
                ]
                for det in detections
            ], dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # 2. ByteTrack update
        track_results = self.tracker.update(
            dets,
        )

        # 3. Domain 모델로 변환
        tracked_objects = []
        for trk in track_results:
            x1, y1, x2, y2 = map(int, trk.xyxy)
            tracked_objects.append(
                TrackedObject(
                    track_id=int(trk.track_id),
                    bbox=(x1, y1, x2, y2),
                    score=float(trk.score),
                    class_id=int(trk.class_id),
                    class_name=str(trk.class_id),
                )
            )

        return tracked_objects
