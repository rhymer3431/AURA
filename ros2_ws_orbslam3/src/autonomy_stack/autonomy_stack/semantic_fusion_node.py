from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import DefaultDict, MutableMapping

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray


def _sensor_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


def _map_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _class_color(class_id: int) -> tuple[float, float, float]:
    r = ((37 * class_id + 97) % 255) / 255.0
    g = ((17 * class_id + 193) % 255) / 255.0
    b = ((29 * class_id + 53) % 255) / 255.0
    return float(r), float(g), float(b)


def _quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _parse_int_set(raw: str) -> set[int]:
    out: set[int] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def _build_xyz_cloud(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.is_bigendian = False
    msg.is_dense = False
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = pts.tobytes()
    return msg


def _build_xyzl_cloud(points_xyz: np.ndarray, labels: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    lbl = np.asarray(labels, dtype=np.float32).reshape(-1)
    count = min(pts.shape[0], lbl.shape[0])
    if pts.shape[0] != count:
        pts = pts[:count]
    if lbl.shape[0] != count:
        lbl = lbl[:count]

    packed = np.zeros((count, 4), dtype=np.float32)
    if count > 0:
        packed[:, :3] = pts
        packed[:, 3] = lbl

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(count)
    msg.is_bigendian = False
    msg.is_dense = False
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="label", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = packed.tobytes()
    return msg


class SemanticFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_fusion_node")
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self.declare_parameter("semantic_topic", "/semantic/label")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("rgb_info_topic", "/camera/rgb/camera_info")
        self.declare_parameter("pose_topic", "/orbslam/pose")
        self.declare_parameter("marker_topic", "/semantic_map/markers")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.15)
        self.declare_parameter("size_x", 20.0)
        self.declare_parameter("size_y", 20.0)
        self.declare_parameter("pixel_stride", 6)
        self.declare_parameter("min_depth", 0.2)
        self.declare_parameter("max_depth", 6.0)
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("max_sync_dt", 0.12)
        self.declare_parameter("max_pose_age", 0.25)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("max_markers", 1200)
        self.declare_parameter("min_cell_votes", 2)
        self.declare_parameter("publish_octomap_cloud", True)
        self.declare_parameter("publish_semantic_cloud", True)
        self.declare_parameter("publish_projected_map", True)
        self.declare_parameter("octomap_cloud_topic", "/semantic_map/octomap_cloud")
        self.declare_parameter("semantic_cloud_topic", "/semantic_map/semantic_cloud")
        self.declare_parameter("projected_map_topic", "/semantic_map/projected_map")
        self.declare_parameter("voxel_resolution", 0.2)
        self.declare_parameter("size_z", 4.0)
        self.declare_parameter("origin_z", -0.5)
        self.declare_parameter("min_voxel_votes", 2)
        self.declare_parameter("max_voxels", 60000)
        self.declare_parameter("max_cloud_points", 20000)
        self.declare_parameter("projection_min_z", 0.05)
        self.declare_parameter("projection_max_z", 1.8)
        self.declare_parameter("octomap_min_z", -0.3)
        self.declare_parameter("octomap_max_z", 2.5)
        self.declare_parameter("free_class_ids", "")
        self.declare_parameter("unlabeled_class_id", 0)
        self.declare_parameter("pose_fallback_identity", False)
        self.declare_parameter("allow_stale_pose", True)

        semantic_topic = self.get_parameter("semantic_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        rgb_info_topic = self.get_parameter("rgb_info_topic").get_parameter_value().string_value
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        self._map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self._resolution = float(self.get_parameter("resolution").value)
        self._size_x = float(self.get_parameter("size_x").value)
        self._size_y = float(self.get_parameter("size_y").value)
        self._stride = max(1, int(self.get_parameter("pixel_stride").value))
        self._min_depth = float(self.get_parameter("min_depth").value)
        self._max_depth = float(self.get_parameter("max_depth").value)
        self._depth_scale = float(self.get_parameter("depth_scale").value)
        self._max_sync_dt = float(self.get_parameter("max_sync_dt").value)
        self._max_pose_age = float(self.get_parameter("max_pose_age").value)
        self._max_markers = int(self.get_parameter("max_markers").value)
        self._min_cell_votes = int(self.get_parameter("min_cell_votes").value)
        self._publish_octomap_cloud = bool(self.get_parameter("publish_octomap_cloud").value)
        self._publish_semantic_cloud = bool(self.get_parameter("publish_semantic_cloud").value)
        self._enable_projected_map = bool(self.get_parameter("publish_projected_map").value)
        octomap_cloud_topic = str(self.get_parameter("octomap_cloud_topic").value)
        semantic_cloud_topic = str(self.get_parameter("semantic_cloud_topic").value)
        projected_map_topic = str(self.get_parameter("projected_map_topic").value)
        self._voxel_resolution = max(0.02, float(self.get_parameter("voxel_resolution").value))
        self._size_z = max(self._voxel_resolution, float(self.get_parameter("size_z").value))
        self._origin_z = float(self.get_parameter("origin_z").value)
        self._min_voxel_votes = max(1, int(self.get_parameter("min_voxel_votes").value))
        self._max_voxels = max(1, int(self.get_parameter("max_voxels").value))
        self._max_cloud_points = max(0, int(self.get_parameter("max_cloud_points").value))
        self._projection_min_z = float(self.get_parameter("projection_min_z").value)
        self._projection_max_z = float(self.get_parameter("projection_max_z").value)
        self._octomap_min_z = float(self.get_parameter("octomap_min_z").value)
        self._octomap_max_z = float(self.get_parameter("octomap_max_z").value)
        self._free_class_ids = _parse_int_set(str(self.get_parameter("free_class_ids").value))
        self._unlabeled_class_id = max(0, int(self.get_parameter("unlabeled_class_id").value))
        self._pose_fallback_identity = bool(self.get_parameter("pose_fallback_identity").value)
        self._allow_stale_pose = bool(self.get_parameter("allow_stale_pose").value)
        self._free_class_ids_np = (
            np.array(sorted(self._free_class_ids), dtype=np.int32)
            if self._free_class_ids
            else np.empty((0,), dtype=np.int32)
        )
        publish_hz = float(self.get_parameter("publish_hz").value)

        self._nx = max(1, int(round(self._size_x / self._resolution)))
        self._ny = max(1, int(round(self._size_y / self._resolution)))
        self._origin_x = -0.5 * self._size_x
        self._origin_y = -0.5 * self._size_y
        self._vx_nx = max(1, int(round(self._size_x / self._voxel_resolution)))
        self._vx_ny = max(1, int(round(self._size_y / self._voxel_resolution)))
        self._vz_nz = max(1, int(round(self._size_z / self._voxel_resolution)))

        self._latest_depth_msg: Image | None = None
        self._latest_depth_t: float = -1.0
        self._latest_pose_msg: PoseStamped | None = None
        self._latest_pose_t: float = -1.0
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None

        self._cell_votes: DefaultDict[tuple[int, int], DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._voxel_votes: DefaultDict[tuple[int, int, int], DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        qos = _sensor_qos(5)
        self.create_subscription(Image, depth_topic, self._on_depth, qos)
        self.create_subscription(CameraInfo, rgb_info_topic, self._on_camera_info, 10)
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 20)
        self.create_subscription(Image, semantic_topic, self._on_semantic, qos)
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, 1)
        self._semantic_cloud_pub = (
            self.create_publisher(PointCloud2, semantic_cloud_topic, _sensor_qos(1))
            if self._publish_semantic_cloud
            else None
        )
        self._octomap_cloud_pub = (
            self.create_publisher(PointCloud2, octomap_cloud_topic, _sensor_qos(1))
            if self._publish_octomap_cloud
            else None
        )
        self._projected_map_pub = (
            self.create_publisher(OccupancyGrid, projected_map_topic, _map_qos())
            if self._enable_projected_map
            else None
        )
        self._publish_timer = self.create_timer(max(0.1, 1.0 / max(0.1, publish_hz)), self._publish_outputs)
        self.get_logger().info(
            "Semantic fusion node started "
            f"(semantic={semantic_topic}, pose={pose_topic}, depth={depth_topic}, octomap_cloud={octomap_cloud_topic})"
        )

    def _accumulate_votes_locked(
        self,
        keys: np.ndarray,
        class_ids: np.ndarray,
        vote_store: MutableMapping[tuple[int, ...], DefaultDict[int, int]],
    ) -> None:
        if keys.shape[0] == 0 or class_ids.shape[0] == 0:
            return

        packed = np.concatenate(
            (
                np.asarray(keys, dtype=np.int32),
                np.asarray(class_ids, dtype=np.int32).reshape(-1, 1),
            ),
            axis=1,
        )
        unique_rows, counts = np.unique(packed, axis=0, return_counts=True)
        for row, count in zip(unique_rows.tolist(), counts.tolist()):
            class_id = int(row[-1])
            key = tuple(int(v) for v in row[:-1])
            vote_store[key][class_id] += int(count)

    def _prune_voxels_locked(self) -> None:
        if len(self._voxel_votes) <= self._max_voxels:
            return

        scored = [
            (int(sum(vote_map.values())), key)
            for key, vote_map in self._voxel_votes.items()
            if vote_map
        ]
        if len(scored) <= self._max_voxels:
            return

        scored.sort(key=lambda item: item[0], reverse=True)
        keep_keys = {key for _, key in scored[: self._max_voxels]}
        for key in list(self._voxel_votes.keys()):
            if key not in keep_keys:
                del self._voxel_votes[key]

    def _snapshot_votes(self) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int, int]]]:
        cells: list[tuple[int, int, int, int]] = []
        voxels: list[tuple[int, int, int, int, int]] = []

        with self._lock:
            for (ix, iy), vote_map in self._cell_votes.items():
                if not vote_map:
                    continue
                class_id, _ = max(vote_map.items(), key=lambda kv: kv[1])
                total_votes = int(sum(vote_map.values()))
                if total_votes < self._min_cell_votes:
                    continue
                cells.append((ix, iy, int(class_id), total_votes))

            for (ix, iy, iz), vote_map in self._voxel_votes.items():
                if not vote_map:
                    continue
                class_id, _ = max(vote_map.items(), key=lambda kv: kv[1])
                total_votes = int(sum(vote_map.values()))
                if total_votes < self._min_voxel_votes:
                    continue
                voxels.append((ix, iy, iz, int(class_id), total_votes))

        return cells, voxels

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._latest_depth_msg = msg
            self._latest_depth_t = _stamp_to_sec(msg.header.stamp)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 9 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self._fx = float(msg.k[0])
            self._fy = float(msg.k[4])
            self._cx = float(msg.k[2])
            self._cy = float(msg.k[5])

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_pose_msg = msg
            self._latest_pose_t = _stamp_to_sec(msg.header.stamp)

    def _decode_depth(self, msg: Image) -> np.ndarray | None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode depth: {exc}")
            return None

        if msg.encoding == "32FC1":
            return np.asarray(depth, dtype=np.float32)
        if msg.encoding == "16UC1":
            return np.asarray(depth, dtype=np.float32) * self._depth_scale
        self.get_logger().warning(f"Unsupported depth encoding: {msg.encoding}")
        return None

    def _on_semantic(self, msg: Image) -> None:
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return

        t_label = _stamp_to_sec(msg.header.stamp)
        with self._lock:
            depth_msg = self._latest_depth_msg
            t_depth = self._latest_depth_t
            pose_msg = self._latest_pose_msg
            t_pose = self._latest_pose_t

        if depth_msg is None:
            return
        if pose_msg is None and not self._pose_fallback_identity:
            return
        if abs(t_depth - t_label) > self._max_sync_dt:
            return

        try:
            label = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode semantic label image: {exc}")
            return
        label = np.asarray(label, dtype=np.uint16)
        if label.ndim != 2:
            return

        depth = self._decode_depth(depth_msg)
        if depth is None:
            return
        if depth.shape[:2] != label.shape[:2]:
            return

        use_pose = pose_msg is not None
        if use_pose and abs(t_pose - t_label) > self._max_pose_age and not self._allow_stale_pose:
            use_pose = False

        if use_pose and pose_msg is not None:
            q = pose_msg.pose.orientation
            R = _quat_to_rotmat(float(q.x), float(q.y), float(q.z), float(q.w))
            t = np.array(
                [
                    float(pose_msg.pose.position.x),
                    float(pose_msg.pose.position.y),
                    float(pose_msg.pose.position.z),
                ],
                dtype=np.float32,
            )
        elif self._pose_fallback_identity:
            R = np.eye(3, dtype=np.float32)
            t = np.zeros((3,), dtype=np.float32)
        else:
            return

        sample_label = label[:: self._stride, :: self._stride]
        sample_depth = depth[:: self._stride, :: self._stride]
        ys, xs = np.where(sample_label > 0)

        if ys.size == 0:
            if self._unlabeled_class_id <= 0:
                return
            depth_valid = np.isfinite(sample_depth) & (sample_depth > self._min_depth) & (sample_depth < self._max_depth)
            ys, xs = np.where(depth_valid)
            if ys.size == 0:
                return
            cls = np.full((ys.size,), self._unlabeled_class_id, dtype=np.int32)
        else:
            cls = sample_label[ys, xs].astype(np.int32)

        u = (xs * self._stride).astype(np.float32)
        v = (ys * self._stride).astype(np.float32)
        z = sample_depth[ys, xs].astype(np.float32)

        valid = np.isfinite(z) & (z > self._min_depth) & (z < self._max_depth)
        if not np.any(valid):
            return
        u = u[valid]
        v = v[valid]
        z = z[valid]
        cls = cls[valid]

        x = (u - self._cx) * z / self._fx
        y = (v - self._cy) * z / self._fy
        pts_cam = np.stack([x, y, z], axis=1)
        pts_map = pts_cam @ R.T + t[None, :]

        cell_x = ((pts_map[:, 0] - self._origin_x) / self._resolution).astype(np.int32)
        cell_y = ((pts_map[:, 1] - self._origin_y) / self._resolution).astype(np.int32)
        cell_in_bounds = (cell_x >= 0) & (cell_x < self._nx) & (cell_y >= 0) & (cell_y < self._ny)

        vox_x = ((pts_map[:, 0] - self._origin_x) / self._voxel_resolution).astype(np.int32)
        vox_y = ((pts_map[:, 1] - self._origin_y) / self._voxel_resolution).astype(np.int32)
        vox_z = ((pts_map[:, 2] - self._origin_z) / self._voxel_resolution).astype(np.int32)
        vox_in_bounds = (
            (vox_x >= 0)
            & (vox_x < self._vx_nx)
            & (vox_y >= 0)
            & (vox_y < self._vx_ny)
            & (vox_z >= 0)
            & (vox_z < self._vz_nz)
        )

        if not np.any(cell_in_bounds) and not np.any(vox_in_bounds):
            return

        with self._lock:
            if np.any(cell_in_bounds):
                cell_keys = np.stack((cell_x[cell_in_bounds], cell_y[cell_in_bounds]), axis=1)
                self._accumulate_votes_locked(cell_keys, cls[cell_in_bounds], self._cell_votes)
            if np.any(vox_in_bounds):
                voxel_keys = np.stack((vox_x[vox_in_bounds], vox_y[vox_in_bounds], vox_z[vox_in_bounds]), axis=1)
                self._accumulate_votes_locked(voxel_keys, cls[vox_in_bounds], self._voxel_votes)
                self._prune_voxels_locked()

    def _publish_outputs(self) -> None:
        cells, voxels = self._snapshot_votes()
        cells.sort(key=lambda item: item[3], reverse=True)
        voxels.sort(key=lambda item: item[4], reverse=True)

        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self._map_frame
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for marker_id, (ix, iy, class_id, votes) in enumerate(cells[: self._max_markers]):
            marker = Marker()
            marker.header.frame_id = self._map_frame
            marker.header.stamp = now
            marker.ns = "semantic_map"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = self._origin_x + (float(ix) + 0.5) * self._resolution
            marker.pose.position.y = self._origin_y + (float(iy) + 0.5) * self._resolution
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = self._resolution
            marker.scale.y = self._resolution
            marker.scale.z = 0.05
            r, g, b = _class_color(class_id)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = min(1.0, 0.25 + 0.75 * (1.0 - math.exp(-0.15 * votes)))
            marker_array.markers.append(marker)

        self._marker_pub.publish(marker_array)

        if self._max_cloud_points > 0 and len(voxels) > self._max_cloud_points:
            voxels = voxels[: self._max_cloud_points]

        if len(voxels) == 0:
            if self._semantic_cloud_pub is not None:
                self._semantic_cloud_pub.publish(
                    _build_xyzl_cloud(np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), self._map_frame, now)
                )
            if self._octomap_cloud_pub is not None:
                self._octomap_cloud_pub.publish(
                    _build_xyz_cloud(np.empty((0, 3), dtype=np.float32), self._map_frame, now)
                )
            if self._projected_map_pub is not None:
                self._publish_projected_map(now, np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32))
            return

        points = np.empty((len(voxels), 3), dtype=np.float32)
        labels = np.empty((len(voxels),), dtype=np.int32)
        for idx, (ix, iy, iz, class_id, _) in enumerate(voxels):
            points[idx, 0] = self._origin_x + (float(ix) + 0.5) * self._voxel_resolution
            points[idx, 1] = self._origin_y + (float(iy) + 0.5) * self._voxel_resolution
            points[idx, 2] = self._origin_z + (float(iz) + 0.5) * self._voxel_resolution
            labels[idx] = int(class_id)

        if self._semantic_cloud_pub is not None:
            self._semantic_cloud_pub.publish(_build_xyzl_cloud(points, labels, self._map_frame, now))

        non_free = np.ones((labels.shape[0],), dtype=bool)
        if self._free_class_ids_np.size > 0:
            non_free = ~np.isin(labels, self._free_class_ids_np)

        octomap_height = (points[:, 2] >= self._octomap_min_z) & (points[:, 2] <= self._octomap_max_z)
        octomap_mask = non_free & octomap_height
        if self._octomap_cloud_pub is not None:
            self._octomap_cloud_pub.publish(_build_xyz_cloud(points[octomap_mask], self._map_frame, now))

        if self._projected_map_pub is not None:
            self._publish_projected_map(now, points, labels)

    def _publish_projected_map(self, stamp, points: np.ndarray, labels: np.ndarray) -> None:
        if self._projected_map_pub is None:
            return

        grid = np.full((self._ny, self._nx), -1, dtype=np.int8)
        if points.shape[0] > 0:
            mx = ((points[:, 0] - self._origin_x) / self._resolution).astype(np.int32)
            my = ((points[:, 1] - self._origin_y) / self._resolution).astype(np.int32)
            in_bounds = (mx >= 0) & (mx < self._nx) & (my >= 0) & (my < self._ny)
            in_height = (points[:, 2] >= self._projection_min_z) & (points[:, 2] <= self._projection_max_z)
            valid = in_bounds & in_height

            if np.any(valid):
                vx = mx[valid]
                vy = my[valid]
                vl = labels[valid]
                if self._free_class_ids_np.size > 0:
                    free_mask = np.isin(vl, self._free_class_ids_np)
                else:
                    free_mask = np.zeros((vl.shape[0],), dtype=bool)

                if np.any(free_mask):
                    grid[vy[free_mask], vx[free_mask]] = 0
                occ_mask = ~free_mask
                if np.any(occ_mask):
                    grid[vy[occ_mask], vx[occ_mask]] = 100

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self._map_frame
        msg.info.map_load_time = stamp
        msg.info.resolution = float(self._resolution)
        msg.info.width = int(self._nx)
        msg.info.height = int(self._ny)
        msg.info.origin.position.x = float(self._origin_x)
        msg.info.origin.position.y = float(self._origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).tolist()
        self._projected_map_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SemanticFusionNode()
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
