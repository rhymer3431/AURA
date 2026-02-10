from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


def _map_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _pose_qos() -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=QoSReliabilityPolicy.RELIABLE,
    )


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


@dataclass
class BlacklistGoal:
    x: float
    y: float
    stamp_sec: float


class FrontierExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__("frontier_explorer_node")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("pose_topic", "/orbslam/pose")
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tick_hz", 1.0)
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("min_frontier_cluster_size", 8)
        self.declare_parameter("min_goal_distance", 0.8)
        self.declare_parameter("max_goal_distance", 8.0)
        self.declare_parameter("goal_timeout_sec", 45.0)
        self.declare_parameter("goal_reached_dist", 0.45)
        self.declare_parameter("blacklist_radius", 0.7)
        self.declare_parameter("blacklist_ttl_sec", 120.0)
        self.declare_parameter("no_frontier_idle_sec", 2.0)

        map_topic = str(self.get_parameter("map_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        action_name = str(self.get_parameter("action_name").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        tick_hz = max(0.2, float(self.get_parameter("tick_hz").value))
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._min_cluster_size = max(1, int(self.get_parameter("min_frontier_cluster_size").value))
        self._min_goal_dist = max(0.1, float(self.get_parameter("min_goal_distance").value))
        self._max_goal_dist = max(self._min_goal_dist, float(self.get_parameter("max_goal_distance").value))
        self._goal_timeout_sec = max(5.0, float(self.get_parameter("goal_timeout_sec").value))
        self._goal_reached_dist = max(0.05, float(self.get_parameter("goal_reached_dist").value))
        self._blacklist_radius = max(0.05, float(self.get_parameter("blacklist_radius").value))
        self._blacklist_ttl_sec = max(1.0, float(self.get_parameter("blacklist_ttl_sec").value))
        self._no_frontier_idle_sec = max(0.2, float(self.get_parameter("no_frontier_idle_sec").value))

        self._map: Optional[OccupancyGrid] = None
        self._pose_xyyaw: Optional[tuple[float, float, float]] = None

        self._action_client = ActionClient(self, NavigateToPose, action_name)
        self._goal_handle = None
        self._result_future = None
        self._goal_sent_sec: Optional[float] = None
        self._current_goal_xy: Optional[tuple[float, float]] = None
        self._pending_goal_xy: Optional[tuple[float, float]] = None
        self._blacklist: list[BlacklistGoal] = []
        self._last_no_frontier_log_sec: float = -1e9

        self.create_subscription(OccupancyGrid, map_topic, self._on_map, _map_qos())
        self.create_subscription(PoseStamped, pose_topic, self._on_pose, _pose_qos())
        self._timer = self.create_timer(1.0 / tick_hz, self._tick)
        self.get_logger().info(
            f"Frontier explorer started (map={map_topic}, pose={pose_topic}, action={action_name})"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        q = msg.pose.orientation
        yaw = _quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self._pose_xyyaw = (x, y, yaw)

    def _prune_blacklist(self) -> None:
        now_sec = self._now_sec()
        self._blacklist = [b for b in self._blacklist if (now_sec - b.stamp_sec) <= self._blacklist_ttl_sec]

    def _is_blacklisted(self, gx: float, gy: float) -> bool:
        for b in self._blacklist:
            if math.hypot(gx - b.x, gy - b.y) <= self._blacklist_radius:
                return True
        return False

    def _add_blacklist(self, gx: float, gy: float) -> None:
        self._blacklist.append(BlacklistGoal(gx, gy, self._now_sec()))

    def _grid_from_map(self, msg: OccupancyGrid) -> Optional[np.ndarray]:
        w = int(msg.info.width)
        h = int(msg.info.height)
        if w <= 0 or h <= 0 or len(msg.data) != w * h:
            return None
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        return arr

    def _neighbor_unknown_mask(self, unknown: np.ndarray) -> np.ndarray:
        h, w = unknown.shape
        padded = np.pad(unknown, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        out = np.zeros((h, w), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                out |= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        return out

    def _choose_frontier_goal(self) -> Optional[tuple[float, float]]:
        if self._map is None or self._pose_xyyaw is None:
            return None
        map_msg = self._map
        grid = self._grid_from_map(map_msg)
        if grid is None:
            return None

        unknown = (grid == -1) | (grid == 255)
        free = grid == 0
        frontier = free & self._neighbor_unknown_mask(unknown)
        if not np.any(frontier):
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return None

        rx, ry, _ = self._pose_xyyaw
        res = float(map_msg.info.resolution)
        ox = float(map_msg.info.origin.position.x)
        oy = float(map_msg.info.origin.position.y)

        best_goal = None
        best_score = -1e9

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._min_cluster_size:
                continue

            cyf, cxf = centroids[label][1], centroids[label][0]
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            # Pick one free frontier cell near cluster centroid.
            d2 = (xs.astype(np.float32) - cxf) ** 2 + (ys.astype(np.float32) - cyf) ** 2
            idx = int(np.argmin(d2))
            gx_cell = int(xs[idx])
            gy_cell = int(ys[idx])

            gx = ox + (float(gx_cell) + 0.5) * res
            gy = oy + (float(gy_cell) + 0.5) * res
            dist = math.hypot(gx - rx, gy - ry)
            if dist < self._min_goal_dist or dist > self._max_goal_dist:
                continue
            if self._is_blacklisted(gx, gy):
                continue

            score = float(area) / (dist + 0.5)
            if score > best_score:
                best_score = score
                best_goal = (gx, gy)

        return best_goal

    def _cancel_current_goal(self, reason: str) -> None:
        if self._goal_handle is None:
            return
        if self._current_goal_xy is not None:
            self._add_blacklist(self._current_goal_xy[0], self._current_goal_xy[1])
        self.get_logger().warn(f"Cancelling exploration goal ({reason}).")
        try:
            self._goal_handle.cancel_goal_async()
        except Exception:
            pass
        self._goal_handle = None
        self._result_future = None
        self._goal_sent_sec = None
        self._current_goal_xy = None

    def _feedback_cb(self, _feedback_msg) -> None:
        # Feedback is optional for this loop.
        return

    def _goal_response_cb(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Goal request failed: {exc}")
            if self._pending_goal_xy is not None:
                self._add_blacklist(self._pending_goal_xy[0], self._pending_goal_xy[1])
            self._pending_goal_xy = None
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Exploration goal rejected.")
            if self._pending_goal_xy is not None:
                self._add_blacklist(self._pending_goal_xy[0], self._pending_goal_xy[1])
            self._pending_goal_xy = None
            return

        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_cb)
        self._goal_sent_sec = self._now_sec()
        self._current_goal_xy = self._pending_goal_xy
        self._pending_goal_xy = None
        if self._current_goal_xy is not None:
            self.get_logger().info(
                f"Exploration goal accepted: ({self._current_goal_xy[0]:.2f}, {self._current_goal_xy[1]:.2f})"
            )

    def _result_cb(self, future) -> None:
        status = GoalStatus.STATUS_UNKNOWN
        try:
            wrapped = future.result()
            status = int(wrapped.status)
        except Exception as exc:
            self.get_logger().warn(f"Failed to get goal result: {exc}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Exploration goal reached.")
        else:
            if self._current_goal_xy is not None:
                self._add_blacklist(self._current_goal_xy[0], self._current_goal_xy[1])
            self.get_logger().warn(f"Exploration goal finished with status={status}.")

        self._goal_handle = None
        self._result_future = None
        self._goal_sent_sec = None
        self._current_goal_xy = None

    def _send_goal(self, gx: float, gy: float) -> None:
        if self._pose_xyyaw is None:
            return
        if not self._action_client.wait_for_server(timeout_sec=0.5):
            return

        rx, ry, _ = self._pose_xyyaw
        yaw = math.atan2(gy - ry, gx - rx)
        qx, qy, qz, qw = _yaw_to_quat(yaw)

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self._map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(gx)
        goal.pose.pose.position.y = float(gy)
        goal.pose.pose.orientation.x = float(qx)
        goal.pose.pose.orientation.y = float(qy)
        goal.pose.pose.orientation.z = float(qz)
        goal.pose.pose.orientation.w = float(qw)

        self._pending_goal_xy = (gx, gy)
        future = self._action_client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        future.add_done_callback(self._goal_response_cb)

    def _tick(self) -> None:
        self._prune_blacklist()

        if self._goal_handle is not None and self._goal_sent_sec is not None:
            elapsed = self._now_sec() - self._goal_sent_sec
            if elapsed > self._goal_timeout_sec:
                self._cancel_current_goal("timeout")
                return

            if self._pose_xyyaw is not None and self._current_goal_xy is not None:
                dist = math.hypot(
                    self._pose_xyyaw[0] - self._current_goal_xy[0],
                    self._pose_xyyaw[1] - self._current_goal_xy[1],
                )
                if dist <= self._goal_reached_dist:
                    self._cancel_current_goal("already near goal")
            return

        if self._pending_goal_xy is not None:
            return

        goal = self._choose_frontier_goal()
        if goal is None:
            now_sec = self._now_sec()
            if now_sec - self._last_no_frontier_log_sec >= self._no_frontier_idle_sec:
                self.get_logger().info("No frontier found. Waiting for map expansion.")
                self._last_no_frontier_log_sec = now_sec
            return

        self._send_goal(goal[0], goal[1])


def main() -> None:
    rclpy.init()
    node = FrontierExplorerNode()
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
