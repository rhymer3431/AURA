import cv2
import time
import threading
import queue
from domain.frame.frame_context import FrameContext
from application.use_cases.process_frame import ProcessFrameUseCase


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
        graph_send_interval: int = 5,
    ):
        self.use_case = use_case
        self.video_input = video_input
        self.visualizer = visualizer
        self.vis_client = vis_client
        self.frame_id = 0
        self.target_fps = target_fps
        self.send_every = max(1, send_every)
        self.jpeg_quality = max(30, min(95, jpeg_quality))  # retained for compatibility
        self.graph_send_interval = max(1, graph_send_interval)

        self.stop_event = threading.Event()
        self.video_queue: queue.Queue = queue.Queue(maxsize=1)
        self.graph_queue: queue.Queue = queue.Queue(maxsize=1)

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

        video_thread = threading.Thread(target=self._video_sender, daemon=True)
        graph_thread = threading.Thread(target=self._graph_sender, daemon=True)
        video_thread.start()
        graph_thread.start()

        while not self.stop_event.is_set():
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

            # Queue video frame for streaming (send_every controls bandwidth)
            if self.vis_client and (self.frame_id % self.send_every == 0):
                # push raw BGR frame; downstream UI converts to texture
                self._offer_queue(self.video_queue, vis_frame.copy())

            # Queue graph every graph_send_interval frames
            if self.vis_client and (self.frame_id % self.graph_send_interval == 0):
                if ctx.scene_graph:
                    self._offer_queue(self.graph_queue, self._graph_to_json(ctx.scene_graph))

            # 로컬 디스플레이 (optional)
            # cv2.imshow("YOLO + ByteTrack", vis_frame)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Frame pacing to target FPS (0 => as fast as possible)
            if frame_interval > 0:
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        video_thread.join(timeout=1)
        graph_thread.join(timeout=1)

    def _offer_queue(self, q: queue.Queue, item):
        try:
            if q.full():
                q.get_nowait()
            q.put_nowait(item)
        except queue.Full:
            pass
        except queue.Empty:
            pass

    def _video_sender(self):
        if not self.vis_client:
            return
        while not self.stop_event.is_set():
            try:
                frame_data = self.video_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self.vis_client.send_frame(frame_data)

    def _graph_sender(self):
        if not self.vis_client:
            return
        while not self.stop_event.is_set():
            try:
                graph_json = self.graph_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.vis_client.send_graph(graph_json)
