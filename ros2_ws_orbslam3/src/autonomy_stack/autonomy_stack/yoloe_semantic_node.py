from __future__ import annotations

import json
import threading
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


def _class_color(class_id: int) -> tuple[int, int, int]:
    # Deterministic RGB palette.
    r = (37 * class_id + 97) % 255
    g = (17 * class_id + 193) % 255
    b = (29 * class_id + 53) % 255
    return int(b), int(g), int(r)


class YoloeSemanticNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_semantic_node")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_rgb: Image | None = None
        self._last_processed_stamp_ns: int = -1

        self.declare_parameter("rgb_topic", "/camera/rgb")
        self.declare_parameter("semantic_topic", "/semantic/label")
        self.declare_parameter("overlay_topic", "/semantic/overlay")
        self.declare_parameter("class_map_topic", "/semantic/class_map")
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_det", 50)
        self.declare_parameter("device", "")
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("overlay_alpha", 0.45)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("infer_hz", 12.0)
        self.declare_parameter("classes", "")
        self.declare_parameter("disable_open_vocab", False)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        semantic_topic = self.get_parameter("semantic_topic").get_parameter_value().string_value
        overlay_topic = self.get_parameter("overlay_topic").get_parameter_value().string_value
        class_map_topic = self.get_parameter("class_map_topic").get_parameter_value().string_value
        requested_model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self._model_path = self._resolve_model_path(requested_model_path)
        self._conf = float(self.get_parameter("conf").value)
        self._iou = float(self.get_parameter("iou").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        self._max_det = int(self.get_parameter("max_det").value)
        self._device = self.get_parameter("device").get_parameter_value().string_value
        self._mask_threshold = float(self.get_parameter("mask_threshold").value)
        self._overlay_alpha = float(self.get_parameter("overlay_alpha").value)
        self._publish_overlay = bool(self.get_parameter("publish_overlay").value)
        raw_classes = str(self.get_parameter("classes").value)
        self._classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        self._disable_open_vocab = bool(self.get_parameter("disable_open_vocab").value)
        infer_hz = float(self.get_parameter("infer_hz").value)

        qos = _sensor_qos(5)
        self._label_pub = self.create_publisher(Image, semantic_topic, qos)
        self._overlay_pub = self.create_publisher(Image, overlay_topic, qos)
        self._class_map_pub = self.create_publisher(String, class_map_topic, 1)
        self.create_subscription(Image, rgb_topic, self._on_rgb, qos)
        self._timer = self.create_timer(max(0.01, 1.0 / max(0.1, infer_hz)), self._infer_once)
        self._class_map_timer = self.create_timer(2.0, self._publish_class_map)

        self._model = self._load_model(self._model_path)
        self._apply_open_vocab_if_possible()
        self._class_map_json = self._build_class_map_json(self._model.names)
        self.get_logger().info(f"YOLOE model loaded: {self._model_path}")

    def _resolve_model_path(self, requested: str) -> str:
        if requested:
            req_path = Path(requested)
            if req_path.exists():
                return str(req_path.resolve())

        project_root = Path(os.getenv("PROJECT_DIR", "")).resolve() if os.getenv("PROJECT_DIR") else None
        if project_root is None or not project_root.exists():
            # .../project/ros2_ws_orbslam3/src/autonomy_stack/autonomy_stack/yoloe_semantic_node.py
            project_root = Path(__file__).resolve().parents[4]

        # Search common local pretrained weights first.
        candidates = [
            project_root / "weights" / "yoloe-26s-seg.pt",
            project_root / "weights" / "yoloe-26s-seg-pf.pt",
            project_root / "weights" / "yoloe-11s-seg.pt",
            project_root / "AURA" / "yoloe-26s-seg.pt",
            project_root / "AURA" / "models" / "yoloe-26s-seg.pt",
        ]

        if requested:
            req_path = Path(requested)
            candidates.insert(0, req_path)
            candidates.insert(1, Path.cwd() / req_path.name)
            candidates.insert(2, Path.cwd() / req_path)

        for path in candidates:
            if path.exists():
                return str(path.resolve())

        if requested:
            return requested
        # Let ultralytics handle its own default behavior if local search fails.
        return "yoloe-26s-seg.pt"

    def _load_model(self, model_path: str) -> Any:
        path = Path(model_path)
        if path.exists():
            model_ref = str(path.resolve())
        else:
            model_ref = model_path
        try:
            from ultralytics import YOLO, YOLOE  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ultralytics is not available. Install with `pip install ultralytics torch`."
            ) from exc
        name_lower = Path(model_ref).name.lower()
        if "yoloe" in name_lower:
            return YOLOE(model_ref)
        return YOLO(model_ref)

    def _apply_open_vocab_if_possible(self) -> None:
        if self._disable_open_vocab:
            self.get_logger().info("Open-vocab disabled by parameter.")
            return
        if not self._classes:
            return
        if not hasattr(self._model, "set_classes"):
            self.get_logger().warning("Model does not support set_classes().")
            return
        try:
            self._model.set_classes(self._classes)
            self.get_logger().info(f"Open-vocab classes applied: {self._classes}")
        except Exception as exc:
            self.get_logger().warning(
                "Failed to apply open-vocab classes; fallback to model default classes. "
                f"reason={exc}"
            )

    def _build_class_map_json(self, names: Any) -> str:
        if isinstance(names, dict):
            items = {str(int(k) + 1): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            items = {str(i + 1): str(v) for i, v in enumerate(names)}
        else:
            items = {}
        return json.dumps(items, ensure_ascii=True)

    def _publish_class_map(self) -> None:
        msg = String()
        msg.data = self._class_map_json
        self._class_map_pub.publish(msg)

    def _on_rgb(self, msg: Image) -> None:
        with self._lock:
            self._latest_rgb = msg

    def _infer_once(self) -> None:
        with self._lock:
            msg = self._latest_rgb
            self._latest_rgb = None
        if msg is None:
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if stamp_ns <= self._last_processed_stamp_ns:
            return
        self._last_processed_stamp_ns = stamp_ns

        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode RGB image: {exc}")
            return

        kwargs: dict[str, Any] = {
            "source": rgb,
            "verbose": False,
            "conf": self._conf,
            "iou": self._iou,
            "imgsz": self._imgsz,
            "max_det": self._max_det,
        }
        if self._device:
            kwargs["device"] = self._device

        try:
            result = self._model.predict(**kwargs)[0]
        except Exception as exc:
            self.get_logger().warning(f"YOLO inference failed: {exc}")
            return

        h, w = rgb.shape[:2]
        label = np.zeros((h, w), dtype=np.uint16)

        if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
            masks = result.masks.data
            boxes_cls = result.boxes.cls
            boxes_conf = result.boxes.conf
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            else:
                masks = np.asarray(masks)
            if hasattr(boxes_cls, "cpu"):
                boxes_cls = boxes_cls.cpu().numpy()
            else:
                boxes_cls = np.asarray(boxes_cls)
            if hasattr(boxes_conf, "cpu"):
                boxes_conf = boxes_conf.cpu().numpy()
            else:
                boxes_conf = np.asarray(boxes_conf)

            order = np.argsort(boxes_conf.astype(np.float32))
            for idx in order:
                class_id = int(boxes_cls[idx]) + 1
                mask = masks[idx] > self._mask_threshold
                label[mask] = np.uint16(class_id)

        label_msg = self._bridge.cv2_to_imgmsg(label, encoding="mono16")
        label_msg.header = msg.header
        self._label_pub.publish(label_msg)

        if not self._publish_overlay:
            return

        overlay = rgb.copy()
        class_ids = np.unique(label)
        class_ids = class_ids[class_ids > 0]
        for class_id in class_ids:
            color = _class_color(int(class_id))
            class_mask = label == class_id
            if not np.any(class_mask):
                continue
            overlay[class_mask] = (
                self._overlay_alpha * np.array(color, dtype=np.float32)
                + (1.0 - self._overlay_alpha) * overlay[class_mask].astype(np.float32)
            ).astype(np.uint8)

        overlay_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self._overlay_pub.publish(overlay_msg)


def main() -> None:
    rclpy.init()
    node: YoloeSemanticNode | None = None
    try:
        node = YoloeSemanticNode()
    except Exception as exc:
        temp_node = rclpy.create_node("yoloe_semantic_node_init_error")
        temp_node.get_logger().error(str(exc))
        temp_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
