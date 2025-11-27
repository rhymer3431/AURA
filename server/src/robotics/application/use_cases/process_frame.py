# src/robotics/application/use_cases/process_frame.py

from typing import Optional


class ProcessFrameUseCase:
    def __init__(self, detector, sg_reasoner: Optional[object] = None):
        self.detector = detector
        self.sg_reasoner = sg_reasoner

    def execute(self, ctx):
        """
        YOLO-World Track 기반 처리:
        - Detection + Tracking ID 수행
        - 결과를 ctx.detections에 넣고 필요 시 Scene Graph까지 생성
        """
        ctx.detections = self.detector.track(ctx.raw_frame)

        if self.sg_reasoner and ctx.has_detections():
            ctx.scene_graph = self.sg_reasoner.infer_from_detections(ctx.detections)

        return ctx
