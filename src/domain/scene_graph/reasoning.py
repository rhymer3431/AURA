from typing import List

from domain.detection.detected_object import DetectedObject
from domain.scene_graph.entities import Node, SceneGraph
from domain.scene_graph.builder import SceneGraphBuilder


def nodes_from_detections(detections: List[DetectedObject]) -> List[Node]:
    """Convert DetectedObject list into SceneGraph nodes."""
    nodes: List[Node] = []
    for idx, det in enumerate(detections):
        node_id = det.track_id if det.track_id is not None else idx
        label = det.class_name
        bbox = list(det.bbox)
        features = {
            "score": det.score,
            "class_id": det.class_id,
            "track_id": det.track_id,
        }
        nodes.append(Node(id=node_id, label=label, bbox=bbox, features=features))
    return nodes


class SceneGraphReasoner:
    """
    Thin wrapper around SceneGraphBuilder to keep domain reasoning isolated.
    """

    def __init__(self, builder: SceneGraphBuilder):
        self.builder = builder

    def infer_from_detections(self, detections: List[DetectedObject]) -> SceneGraph:
        if not detections:
            return SceneGraph()
        nodes = nodes_from_detections(detections)
        return self.builder.build(nodes)
