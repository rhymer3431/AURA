import asyncio
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class VideoFileFrameSource:
    """Looping video-file frame source."""

    is_live = False

    def __init__(self, video_path: Path | str) -> None:
        self.video_path = str(video_path)
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_path}")

    async def read_frame(self):
        return await asyncio.to_thread(self._read_frame_blocking)

    async def close(self) -> None:
        await asyncio.to_thread(self._release_blocking)

    def get_live_metadata(self) -> Dict[str, Any]:
        return {
            "streamSource": {
                "type": "video",
                "videoPath": self.video_path,
            }
        }

    def _read_frame_blocking(self):
        while True:
            ok, frame_bgr = self._cap.read()
            if ok:
                return frame_bgr
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _release_blocking(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class Ros2ImageTopicFrameSource:
    """ROS2 sensor_msgs/Image subscriber frame source."""

    is_live = True

    def __init__(
        self,
        image_topic: str = "/image_raw",
        queue_size: int = 1,
        slam_pose_topic: Optional[str] = "/orbslam/pose",
        semantic_projected_map_topic: Optional[str] = "/semantic_map/projected_map",
        semantic_octomap_cloud_topic: Optional[str] = "/semantic_map/octomap_cloud",
    ) -> None:
        self.image_topic = image_topic
        self.queue_size = max(1, int(queue_size))
        self.slam_pose_topic = (
            slam_pose_topic.strip() if isinstance(slam_pose_topic, str) else None
        )
        self.semantic_projected_map_topic = (
            semantic_projected_map_topic.strip()
            if isinstance(semantic_projected_map_topic, str)
            and semantic_projected_map_topic.strip()
            else None
        )
        self.semantic_octomap_cloud_topic = (
            semantic_octomap_cloud_topic.strip()
            if isinstance(semantic_octomap_cloud_topic, str)
            and semantic_octomap_cloud_topic.strip()
            else None
        )
        self._frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._pose_lock = threading.Lock()
        self._semantic_lock = threading.Lock()
        self._latest_slam_pose: Optional[Dict[str, Any]] = None
        self._latest_projected_map_stats: Optional[Dict[str, Any]] = None
        self._latest_projected_map_preview: Optional[Dict[str, Any]] = None
        self._latest_octomap_cloud_stats: Optional[Dict[str, Any]] = None
        self._projected_map_revision: int = 0
        self._projected_map_last_emitted_revision: int = -1

        try:
            from cv_bridge import CvBridge
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import OccupancyGrid
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, PointCloud2
        except Exception as exc:  # pragma: no cover - import guard for non-ROS envs
            raise RuntimeError(
                "ROS2 Python packages are required. Source your ROS2 environment first."
            ) from exc

        self._context = Context()
        self._context.init(args=None)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._bridge = CvBridge()
        node_name = f"aura_stream_input_{uuid.uuid4().hex[:8]}"
        self._node = Node(node_name, context=self._context)

        self._node.create_subscription(
            Image,
            self.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        if self.slam_pose_topic:
            self._node.create_subscription(
                PoseStamped,
                self.slam_pose_topic,
                self._on_slam_pose,
                qos_profile_sensor_data,
            )
        if self.semantic_projected_map_topic:
            self._node.create_subscription(
                OccupancyGrid,
                self.semantic_projected_map_topic,
                self._on_projected_map,
                qos_profile_sensor_data,
            )
        if self.semantic_octomap_cloud_topic:
            self._node.create_subscription(
                PointCloud2,
                self.semantic_octomap_cloud_topic,
                self._on_octomap_cloud,
                qos_profile_sensor_data,
            )
        self._executor.add_node(self._node)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    async def read_frame(self) -> Optional[object]:
        return await asyncio.to_thread(self._read_frame_blocking)

    async def close(self) -> None:
        self._stop_event.set()
        await asyncio.to_thread(self._join_spin_thread)
        self._shutdown_ros()

    def get_live_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "streamSource": {
                "type": "ros2",
                "imageTopic": self.image_topic,
            },
            "slamPoseTopic": self.slam_pose_topic,
            "semanticMap": {
                "projectedMapTopic": self.semantic_projected_map_topic,
                "octomapCloudTopic": self.semantic_octomap_cloud_topic,
            },
        }

        with self._pose_lock:
            if self._latest_slam_pose is not None:
                metadata["slamPose"] = self._latest_slam_pose
        with self._semantic_lock:
            semantic_map = metadata["semanticMap"]
            if self._latest_projected_map_stats is not None:
                projected_map = dict(self._latest_projected_map_stats)
                preview = self._latest_projected_map_preview
                if preview is not None:
                    revision = int(preview.get("revision", -1))
                    projected_map["previewRevision"] = revision
                    if revision != self._projected_map_last_emitted_revision:
                        projected_map["preview"] = preview
                        self._projected_map_last_emitted_revision = revision
                semantic_map["projectedMap"] = projected_map
            if self._latest_octomap_cloud_stats is not None:
                semantic_map["octomapCloud"] = self._latest_octomap_cloud_stats

        return metadata

    def _on_image(self, msg) -> None:
        try:
            frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _on_slam_pose(self, msg) -> None:
        pose = {
            "position": {
                "x": float(msg.pose.position.x),
                "y": float(msg.pose.position.y),
                "z": float(msg.pose.position.z),
            },
            "orientation": {
                "x": float(msg.pose.orientation.x),
                "y": float(msg.pose.orientation.y),
                "z": float(msg.pose.orientation.z),
                "w": float(msg.pose.orientation.w),
            },
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frameId": str(msg.header.frame_id),
        }
        with self._pose_lock:
            self._latest_slam_pose = pose

    def _on_projected_map(self, msg) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        total_cells = width * height
        if total_cells <= 0:
            return

        data = np.asarray(msg.data, dtype=np.int16)
        if data.size != total_cells:
            return
        grid = data.reshape((height, width))

        occupied_cells = int(np.count_nonzero(data >= 50))
        free_cells = int(np.count_nonzero(data == 0))
        unknown_cells = int(np.count_nonzero(data < 0))
        known_cells = occupied_cells + free_cells
        known_ratio = float(known_cells / total_cells) if total_cells > 0 else 0.0
        preview = self._build_projected_map_preview(grid)

        self._projected_map_revision += 1
        revision = int(self._projected_map_revision)
        stats = {
            "width": width,
            "height": height,
            "resolution": float(msg.info.resolution),
            "occupiedCells": occupied_cells,
            "freeCells": free_cells,
            "unknownCells": unknown_cells,
            "knownRatio": known_ratio,
            "frameId": str(msg.header.frame_id),
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "previewRevision": revision,
        }
        preview["revision"] = revision
        with self._semantic_lock:
            self._latest_projected_map_stats = stats
            self._latest_projected_map_preview = preview

    def _on_octomap_cloud(self, msg) -> None:
        stats = {
            "pointCount": int(msg.width) * int(msg.height),
            "frameId": str(msg.header.frame_id),
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        }
        with self._semantic_lock:
            self._latest_octomap_cloud_stats = stats

    def _build_projected_map_preview(self, grid: np.ndarray, max_size: int = 96) -> Dict[str, Any]:
        grid_2d = np.asarray(grid, dtype=np.int16)
        if grid_2d.ndim != 2:
            return {
                "encoding": "ufo-v1",
                "width": 0,
                "height": 0,
                "rows": [],
            }

        src_h, src_w = grid_2d.shape
        out_w = max(1, min(int(src_w), int(max_size)))
        out_h = max(1, min(int(src_h), int(max_size)))

        x_idx = np.linspace(0, src_w - 1, out_w, dtype=np.int32)
        y_idx = np.linspace(0, src_h - 1, out_h, dtype=np.int32)
        sampled = grid_2d[np.ix_(y_idx, x_idx)]

        # 0=unknown, 1=free, 2=occupied
        classes = np.zeros((out_h, out_w), dtype=np.uint8)
        classes[sampled == 0] = 1
        classes[sampled >= 50] = 2

        lut = np.array([ord("u"), ord("f"), ord("o")], dtype=np.uint8)
        encoded = lut[classes]
        rows = [bytes(encoded[row_idx]).decode("ascii") for row_idx in range(out_h)]

        return {
            "encoding": "ufo-v1",
            "width": int(out_w),
            "height": int(out_h),
            "rows": rows,
        }

    def _spin(self) -> None:
        while not self._stop_event.is_set() and self._context.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                if self._stop_event.is_set():
                    break

    def _read_frame_blocking(self) -> Optional[object]:
        if self._stop_event.is_set():
            return None

        try:
            return self._frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def _join_spin_thread(self) -> None:
        if self._spin_thread.is_alive():
            self._spin_thread.join()

    def _shutdown_ros(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        if self._context is not None:
            try:
                if self._context.ok():
                    self._context.shutdown()
            except Exception:
                pass
            self._context = None
