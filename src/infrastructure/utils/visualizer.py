# src/infrastructure/utils/visualizer.py
import cv2
from domain.detect.entity.detection import DetectedObject


class OpenCVOverlayVisualizer:
    def __init__(self, show_track_id=True, show_score=True):
        self.show_track_id = show_track_id
        self.show_score = show_score

    def draw(self, frame, detections: list[DetectedObject], scene_graph=None):
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if self.show_score:
                label += f" {det.score:.2f}"
            if self.show_track_id and det.track_id is not None:
                label = f"ID:{det.track_id} | " + label

            cv2.putText(
                vis, label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

        if scene_graph and getattr(scene_graph, "edges", None):
            for edge in scene_graph.edges:
                cv2.putText(
                    vis,
                    f"{edge.source}->{edge.target}:{edge.relation}",
                    (10, 20 + 15 * (scene_graph.edges.index(edge) % 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                )

        return vis
