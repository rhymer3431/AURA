# src/robotics/application/use_cases/process_frame.py

from typing import Optional
from robotics.domain.detection.detected_object import DetectedObject
from robotics.domain.tracking.tracked_object import TrackedObject

class ProcessFrameUseCase:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker

    def execute(self, ctx):
        # 반드시 raw_frame 사용
        frame = ctx.raw_frame

        # 1) Detection
        detections = self.detector.detect(frame)
        ctx.detections = detections

        # 2) Tracking
        tracks = self.tracker.track(frame, detections)
        ctx.tracked_objects = tracks

        return ctx
