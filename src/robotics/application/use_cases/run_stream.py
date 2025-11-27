# src/robotics/application/use_cases/run_stream.py

import cv2
from robotics.application.dto import FrameContext
from robotics.application.use_cases.process_frame import ProcessFrameUseCase


class VideoStreamRunner:
    def __init__(
        self,
        use_case: ProcessFrameUseCase,
        video_input: str | int,
        visualizer=None
    ):
        self.use_case = use_case
        self.video_input = video_input
        self.visualizer = visualizer

    def run(self):
        cap = cv2.VideoCapture(self.video_input)

        if not cap.isOpened():
            print(f"Failed to open video input: {self.video_input}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) DTO 생성
            ctx = FrameContext(frame_bgr=frame)

            # 2) 유즈케이스 실행 → detection + tracking
            ctx = self.use_case.execute(ctx)

            # 3) 시각화
            if self.visualizer:
                vis_frame = self.visualizer.draw(ctx.frame_bgr, ctx.detections)
                cv2.imshow("YOLO + ByteTrack", vis_frame)
            else:
                cv2.imshow("YOLO + ByteTrack", frame)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
