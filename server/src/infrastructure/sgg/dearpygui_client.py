import queue


class UiBridge:
    """
    Thread-safe bridge for frames and scene-graphs destined for the local UI.
    Uses bounded queues to drop stale items and keep latency low.
    """

    def __init__(self, max_frames: int = 1, max_graphs: int = 1):
        self.frame_q = queue.Queue(maxsize=max_frames)
        self.graph_q = queue.Queue(maxsize=max_graphs)

    def push_frame(self, frame_bgr):
        try:
            if self.frame_q.full():
                self.frame_q.get_nowait()
            self.frame_q.put_nowait(frame_bgr)
        except (queue.Full, queue.Empty):
            pass

    def push_graph(self, graph_json):
        try:
            if self.graph_q.full():
                self.graph_q.get_nowait()
            self.graph_q.put_nowait(graph_json)
        except (queue.Full, queue.Empty):
            pass


class DearPyGuiClient:
    """
    Drop-in replacement for the old HTTP VisClient.
    Directly pushes frames/graphs into the UiBridge queues.
    """

    def __init__(self, ui_bridge: UiBridge):
        self.ui_bridge = ui_bridge

    def send_frame(self, frame_bgr):
        self.ui_bridge.push_frame(frame_bgr)
        return True

    def send_graph(self, graph: dict):
        self.ui_bridge.push_graph(graph)
        return True
