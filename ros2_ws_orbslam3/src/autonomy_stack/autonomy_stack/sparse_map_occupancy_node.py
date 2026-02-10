from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField


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


def _point_field_offset(msg: PointCloud2, name: str) -> Optional[int]:
    for field in msg.fields:
        if field.name == name:
            if field.datatype != PointField.FLOAT32:
                return None
            return int(field.offset)
    return None


def _pointcloud_xyz(msg: PointCloud2) -> np.ndarray:
    point_count = int(msg.width) * int(msg.height)
    if point_count <= 0 or int(msg.point_step) <= 0 or not msg.data:
        return np.empty((0, 3), dtype=np.float32)

    ox = _point_field_offset(msg, "x")
    oy = _point_field_offset(msg, "y")
    oz = _point_field_offset(msg, "z")
    if ox is None or oy is None or oz is None:
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(msg.point_step)
    if max(ox, oy, oz) + 4 > point_step:
        return np.empty((0, 3), dtype=np.float32)

    endian = ">" if bool(msg.is_bigendian) else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [ox, oy, oz],
            "itemsize": point_step,
        }
    )

    width = int(msg.width)
    height = int(msg.height)
    row_step = int(msg.row_step)

    if row_step == width * point_step:
        arr = np.frombuffer(msg.data, dtype=dtype, count=point_count)
    else:
        rows = []
        for row in range(height):
            offset = row * row_step
            rows.append(np.frombuffer(msg.data, dtype=dtype, count=width, offset=offset))
        arr = np.concatenate(rows, axis=0) if rows else np.empty((0,), dtype=dtype)

    xyz = np.empty((arr.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = arr["x"]
    xyz[:, 1] = arr["y"]
    xyz[:, 2] = arr["z"]
    return xyz


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


class SparseMapOccupancyNode(Node):
    def __init__(self) -> None:
        super().__init__("sparse_map_occupancy_node")
        self._lock = threading.Lock()

        self.declare_parameter("map_points_topic", "/orbslam/map_points")
        self.declare_parameter("pose_topic", "/orbslam/pose")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("size_x", 40.0)
        self.declare_parameter("size_y", 40.0)
        self.declare_parameter("min_z", 0.05)
        self.declare_parameter("max_z", 1.5)
        self.declare_parameter("point_stride", 1)
        self.declare_parameter("max_points_per_cloud", 1200)
        self.declare_parameter("max_raytrace_cells", 500)
        self.declare_parameter("raytrace_max_range", 10.0)
        self.declare_parameter("robot_clear_radius", 0.25)
        self.declare_parameter("hit_log_odds", 0.85)
        self.declare_parameter("free_log_odds", 0.40)
        self.declare_parameter("min_log_odds", -3.5)
        self.declare_parameter("max_log_odds", 4.0)
        self.declare_parameter("occupied_probability_threshold", 0.65)
        self.declare_parameter("free_probability_threshold", 0.35)
        self.declare_parameter("pose_timeout", 0.8)
        self.declare_parameter("publish_hz", 2.0)

        map_points_topic = str(self.get_parameter("map_points_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        map_topic = str(self.get_parameter("map_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._resolution = max(0.02, float(self.get_parameter("resolution").value))
        self._size_x = max(self._resolution, float(self.get_parameter("size_x").value))
        self._size_y = max(self._resolution, float(self.get_parameter("size_y").value))
        self._min_z = float(self.get_parameter("min_z").value)
        self._max_z = float(self.get_parameter("max_z").value)
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._max_points_per_cloud = max(0, int(self.get_parameter("max_points_per_cloud").value))
        self._max_raytrace_cells = max(0, int(self.get_parameter("max_raytrace_cells").value))
        self._raytrace_max_range = max(0.0, float(self.get_parameter("raytrace_max_range").value))
        self._robot_clear_radius = max(0.0, float(self.get_parameter("robot_clear_radius").value))
        self._hit_lo = float(self.get_parameter("hit_log_odds").value)
        self._free_lo = max(0.01, float(self.get_parameter("free_log_odds").value))
        self._min_lo = float(self.get_parameter("min_log_odds").value)
        self._max_lo = float(self.get_parameter("max_log_odds").value)
        self._occ_prob_th = float(self.get_parameter("occupied_probability_threshold").value)
        self._free_prob_th = float(self.get_parameter("free_probability_threshold").value)
        self._pose_timeout = max(0.05, float(self.get_parameter("pose_timeout").value))
        publish_hz = max(0.2, float(self.get_parameter("publish_hz").value))

        self._nx = max(1, int(round(self._size_x / self._resolution)))
        self._ny = max(1, int(round(self._size_y / self._resolution)))
        self._origin_x = -0.5 * self._size_x
        self._origin_y = -0.5 * self._size_y
        self._log_odds = np.zeros((self._ny, self._nx), dtype=np.float32)

        clear_cells = int(math.ceil(self._robot_clear_radius / self._resolution))
        self._clear_offsets: list[tuple[int, int]] = []
        if clear_cells > 0:
            radius_sq = clear_cells * clear_cells
            for dy in range(-clear_cells, clear_cells + 1):
                for dx in range(-clear_cells, clear_cells + 1):
                    if dx * dx + dy * dy <= radius_sq:
                        self._clear_offsets.append((dy, dx))

        self._pose_xy: Optional[tuple[float, float]] = None
        self._pose_stamp_sec: float = -1.0

        self.create_subscription(PointCloud2, map_points_topic, self._on_sparse_map, _sensor_qos(5))
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, _sensor_qos(20))
        self._map_pub = self.create_publisher(OccupancyGrid, map_topic, _map_qos())
        self._publish_timer = self.create_timer(1.0 / publish_hz, self._publish_map)

        self.get_logger().info(
            f"Sparse occupancy mapper started: cloud={map_points_topic}, pose={pose_topic}, map={map_topic}"
        )

    def _world_to_grid(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = ((x - self._origin_x) / self._resolution).astype(np.int32)
        gy = ((y - self._origin_y) / self._resolution).astype(np.int32)
        return gx, gy

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._pose_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
            self._pose_stamp_sec = _stamp_to_sec(msg.header.stamp)

    def _on_sparse_map(self, msg: PointCloud2) -> None:
        points = _pointcloud_xyz(msg)
        if points.shape[0] == 0:
            return

        if self._point_stride > 1:
            points = points[:: self._point_stride]

        if self._max_points_per_cloud > 0 and points.shape[0] > self._max_points_per_cloud:
            sel = np.linspace(0, points.shape[0] - 1, self._max_points_per_cloud, dtype=np.int32)
            points = points[sel]

        valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
        valid &= (points[:, 2] >= self._min_z) & (points[:, 2] <= self._max_z)
        if not np.any(valid):
            return
        points = points[valid]

        pose_xy = None
        pose_fresh = False
        pose_sec = -1.0
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            pose_xy = self._pose_xy
            pose_sec = self._pose_stamp_sec
        if pose_xy is not None and pose_sec > 0.0 and (now_sec - pose_sec) <= self._pose_timeout:
            pose_fresh = True

        if pose_fresh and self._raytrace_max_range > 0.0:
            dx = points[:, 0] - pose_xy[0]
            dy = points[:, 1] - pose_xy[1]
            rng_ok = (dx * dx + dy * dy) <= (self._raytrace_max_range * self._raytrace_max_range)
            if not np.any(rng_ok):
                return
            points = points[rng_ok]

        gx, gy = self._world_to_grid(points[:, 0], points[:, 1])
        in_bounds = (gx >= 0) & (gx < self._nx) & (gy >= 0) & (gy < self._ny)
        if not np.any(in_bounds):
            return

        gx = gx[in_bounds]
        gy = gy[in_bounds]
        cells = np.stack((gy, gx), axis=1)
        unique_cells = np.unique(cells, axis=0)
        if unique_cells.shape[0] == 0:
            return

        with self._lock:
            uy = unique_cells[:, 0]
            ux = unique_cells[:, 1]
            self._log_odds[uy, ux] = np.clip(self._log_odds[uy, ux] + self._hit_lo, self._min_lo, self._max_lo)

            if pose_fresh and pose_xy is not None:
                rbx = int((pose_xy[0] - self._origin_x) / self._resolution)
                rby = int((pose_xy[1] - self._origin_y) / self._resolution)
                if 0 <= rbx < self._nx and 0 <= rby < self._ny:
                    ray_cells = unique_cells
                    if self._max_raytrace_cells > 0 and unique_cells.shape[0] > self._max_raytrace_cells:
                        sel = np.linspace(
                            0,
                            unique_cells.shape[0] - 1,
                            self._max_raytrace_cells,
                            dtype=np.int32,
                        )
                        ray_cells = unique_cells[sel]

                    for c in ray_cells:
                        cy = int(c[0])
                        cx = int(c[1])
                        line = _bresenham(rbx, rby, cx, cy)
                        if len(line) <= 1:
                            continue
                        for lx, ly in line[:-1]:
                            if 0 <= lx < self._nx and 0 <= ly < self._ny:
                                self._log_odds[ly, lx] = max(
                                    self._min_lo, self._log_odds[ly, lx] - self._free_lo
                                )

                    if self._clear_offsets:
                        clear_limit = -2.0 * self._free_lo
                        for dy, dx in self._clear_offsets:
                            cx = rbx + dx
                            cy = rby + dy
                            if 0 <= cx < self._nx and 0 <= cy < self._ny:
                                self._log_odds[cy, cx] = min(self._log_odds[cy, cx], clear_limit)

    def _publish_map(self) -> None:
        with self._lock:
            lo = self._log_odds.copy()

        prob = 1.0 / (1.0 + np.exp(-lo))
        grid = np.full(lo.shape, -1, dtype=np.int8)
        grid[prob >= self._occ_prob_th] = 100
        grid[prob <= self._free_prob_th] = 0

        now = self.get_clock().now().to_msg()
        msg = OccupancyGrid()
        msg.header.stamp = now
        msg.header.frame_id = self._map_frame
        msg.info.map_load_time = now
        msg.info.resolution = float(self._resolution)
        msg.info.width = int(self._nx)
        msg.info.height = int(self._ny)
        msg.info.origin.position.x = float(self._origin_x)
        msg.info.origin.position.y = float(self._origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).tolist()
        self._map_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SparseMapOccupancyNode()
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
