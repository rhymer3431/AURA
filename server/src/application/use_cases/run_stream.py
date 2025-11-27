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
        target_fps: int = 20,
        send_every: int = 1,
        jpeg_quality: int = 70,
    ):
        self.use_case = use_case
        self.video_input = video_input
        self.visualizer = visualizer
        self.vis_client = vis_client
        self.frame_id = 0
        self.target_fps = target_fps
        self.send_every = max(1, send_every)
        self.jpeg_quality = max(30, min(95, jpeg_quality))

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

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0

        while True:
            loop_start = time.time()
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
            else:
                vis_frame = frame

            # Stream to web UI server (frame + scene graph)
            if self.vis_client and (self.frame_id % self.send_every == 0):
                ok, buf = cv2.imencode(
                    ".jpg",
                    vis_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    self.vis_client.send_frame(buf.tobytes())
                if ctx.scene_graph:
                    self.vis_client.send_graph(self._graph_to_json(ctx.scene_graph))

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Frame pacing to target FPS
            if frame_interval > 0:
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()
        cv2.destroyAllWindows()
