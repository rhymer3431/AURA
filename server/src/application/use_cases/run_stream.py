import cv2
import time
from server.src.domain.frame.frame_context import FrameContext
from server.src.application.use_cases.process_frame import ProcessFrameUseCase


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
        self.frame_id = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_input)

        if not cap.isOpened():
            print(f"Failed to open video input: {self.video_input}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()

            # FrameContext 생성 (도메인 모델)
            ctx = FrameContext(
                frame_id=self.frame_id,
                timestamp=timestamp,
                raw_frame=frame
            )
            self.frame_id += 1

            # 유즈케이스 실행: detection + tracking (+ scene graph)
            ctx = self.use_case.execute(ctx)

            # 시각화
            if self.visualizer:
                vis_frame = self.visualizer.draw(ctx.raw_frame, ctx.detections, ctx.scene_graph)
                cv2.imshow("YOLO + ByteTrack", vis_frame)
            else:
                cv2.imshow("YOLO + ByteTrack", frame)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
