import cv2
import time
from server.src.domain.frame.frame_context import FrameContext
from server.src.application.use_cases.process_frame import ProcessFrameUseCase


class VideoStreamRunner:
    def __init__(
        self,
        use_case: ProcessFrameUseCase,
        video_input: str | int,
        visualizer=None,
        vis_client=None,
    ):
        self.use_case = use_case
        self.video_input = video_input
        self.visualizer = visualizer
        self.vis_client = vis_client
        self.frame_id = 0

    def _graph_to_json(self, scene_graph):
        if not scene_graph:
            return {"nodes": [], "edges": []}
        nodes = [{"id": str(n.id), "label": n.label} for n in scene_graph.nodes]
        edges = [
            {"source": str(e.source), "target": str(e.target), "label": e.relation}
            for e in scene_graph.edges
        ]
        return {"nodes": nodes, "edges": edges}

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
                vis_frame = self.visualizer.draw(
                    ctx.raw_frame, ctx.detections, ctx.scene_graph
                )
                cv2.imshow("YOLO + ByteTrack", vis_frame)
            else:
                vis_frame = frame
                cv2.imshow("YOLO + ByteTrack", frame)

            # Stream to web UI server (frame + scene graph)
            if self.vis_client:
                ok, buf = cv2.imencode(".jpg", vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    self.vis_client.send_frame(buf.tobytes())
                if ctx.scene_graph:
                    self.vis_client.send_graph(self._graph_to_json(ctx.scene_graph))

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
