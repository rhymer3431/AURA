# src/robotics/application/use_cases/process_frame.py

class ProcessFrameUseCase:
    def __init__(self, detector):
        self.detector = detector

    def execute(self, ctx):
        """
        YOLO-World Track 기반 처리:
        - Detection + Tracking ID 통합 수행
        - 결과를 ctx.detections에 넣어 상위 모듈이 활용
        """
        ctx.detections = self.detector.track(ctx.raw_frame)

        return ctx
