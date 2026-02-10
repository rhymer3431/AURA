import json
import threading
from typing import Dict, List, Optional, Tuple

from src.infrastructure.perception.perception_service_adapter import PerceptionServiceAdapter
from src.utils.streaming.serializer import (
    serialize_frame_entities,
    serialize_relations,
    serialize_sg_diff,
)
from src.utils.box_iou_xyxy import box_iou_xyxy

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
except Exception as exc:  # pragma: no cover - import guard for non-ROS envs
    raise RuntimeError(
        "ROS2 Python packages are required. Source your ROS2 environment "
        "and ensure rclpy, sensor_msgs, std_msgs, and cv_bridge are available."
    ) from exc


class Ros2PerceptionNode(Node):
    def __init__(
        self,
        perception: PerceptionServiceAdapter,
        scene_planner=None,
        image_topic: str = "/image_raw",
        metadata_topic: Optional[str] = "/aura/perception/metadata",
        target_fps: float = 15.0,
        max_entities: int = 16,
    ) -> None:
        super().__init__("aura_perception")
        self.perception = perception
        self.scene_planner = scene_planner
        self.max_entities = max_entities
        self.bridge = CvBridge()

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_stamp: Optional[Tuple[int, int]] = None
        self._last_processed_stamp: Optional[Tuple[int, int]] = None

        self._frame_idx = 0
        self._last_entities: Dict[int, List[float]] = {}
        self._last_caption = ""
        self._last_focus_targets: List[str] = []

        self._process_on_message = target_fps <= 0
        if not self._process_on_message:
            period = 1.0 / float(target_fps)
            self.create_timer(period, self._on_timer)

        self._image_sub = self.create_subscription(
            Image, image_topic, self._on_image, qos_profile_sensor_data
        )

        self._metadata_pub = None
        if metadata_topic:
            qos_profile = QoSProfile(depth=1)
            self._metadata_pub = self.create_publisher(String, metadata_topic, qos_profile)

    def _on_image(self, msg: Image) -> None:
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        with self._frame_lock:
            self._latest_frame = frame_bgr
            self._latest_stamp = stamp

        if self._process_on_message:
            self._process_latest()

    def _on_timer(self) -> None:
        self._process_latest()

    def _process_latest(self) -> None:
        with self._frame_lock:
            if self._latest_frame is None:
                return
            stamp = self._latest_stamp
            if stamp is not None and stamp == self._last_processed_stamp:
                return
            frame_bgr = self._latest_frame.copy()
            self._last_processed_stamp = stamp

        self._frame_idx += 1
        run_grin = self._frame_idx % 10 == 0

        sg_frame, sg_diff, _ = self.perception.process_frame(
            frame_bgr, self._frame_idx, run_grin, max_entities=self.max_entities
        )

        current_entities = {n.entity_id: n.box for n in sg_frame.nodes}
        change_detected = _detect_change(current_entities, self._last_entities)

        if change_detected and sg_frame.nodes and self.scene_planner is not None:
            tensor_sg = self.perception.build_scene_graph_tensor_frame(sg_frame)
            self.scene_planner.submit(self._frame_idx, tensor_sg)

        if self.scene_planner is not None:
            for _, plan in self.scene_planner.poll_results():
                caption = plan.get("caption", "")
                focus_targets = plan.get("focus_targets", [])
                if caption:
                    self._last_caption = caption
                if isinstance(focus_targets, list):
                    self._last_focus_targets = [str(x) for x in focus_targets]
                    self.perception.update_focus_classes(self._last_focus_targets)

        self._last_entities = current_entities

        if self._metadata_pub is None:
            return

        metadata = {
            "type": "metadata",
            "frameIdx": self._frame_idx,
            "caption": self._last_caption,
            "focusTargets": self._last_focus_targets,
            "entities": serialize_frame_entities(sg_frame),
            "entityRecords": self.perception.ltm_entities,
        }

        has_graph_diff = sg_diff is not None and any(len(v) > 0 for v in sg_diff.values())
        should_send_scene_graph = sg_diff is None or has_graph_diff

        if should_send_scene_graph:
            relation_name_map = getattr(self.perception, "RELATION_ID_TO_NAME", {})
            metadata["relations"] = serialize_relations(sg_frame, relation_name_map)
            if has_graph_diff:
                metadata["sceneGraphDiff"] = serialize_sg_diff(sg_diff, relation_name_map)

        payload = json.dumps(metadata)
        self._metadata_pub.publish(String(data=payload))



def _detect_change(
    current_entities: Dict[int, List[float]], last_entities: Dict[int, List[float]]
) -> bool:
    if len(current_entities) != len(last_entities):
        return True
    if set(current_entities.keys()) != set(last_entities.keys()):
        return True
    for eid, box in current_entities.items():
        if box_iou_xyxy(box, last_entities[eid]) < 0.5:
            return True
    return False
