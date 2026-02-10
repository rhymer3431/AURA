from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo

from .params import NavigatorParams, declare_parameters, params_from_node
from .ros_io import sensor_qos, CameraInfoPublisher
from .tf_manager import TFManager, TRACKING_OK
from .depth_costmap import DepthCostmap, CostmapResult
from .planner_astar import AStarPlanner
from .local_controller import LocalController
from .recovery import RecoveryManager


class NavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__("navigator_node")
        base_params = NavigatorParams()
        declare_parameters(self, base_params)
        self._p = params_from_node(self, base_params)

        self._tf = TFManager(self, self._p.tf, self._p.frames)
        self._planner = AStarPlanner(self._p.planner)
        self._controller = LocalController(self._p.controller)
        self._recovery = RecoveryManager(self._p.recovery)

        self._cmd_pub = self.create_publisher(Twist, self._p.topics.cmd_vel, 10)
        self._costmap_pub = self.create_publisher(OccupancyGrid, self._p.topics.costmap, 1)
        self._path_pub = self.create_publisher(Path, self._p.topics.global_path, 1)
        self._status_pub = self.create_publisher(String, self._p.topics.nav_status, 10)

        self.create_subscription(Image, self._p.topics.rgb, self._rgb_cb, sensor_qos())
        self.create_subscription(Image, self._p.topics.depth, self._depth_cb, sensor_qos())
        self.create_subscription(CameraInfo, self._p.topics.depth_info, self._camera_info_cb, 10)
        self.create_subscription(PoseStamped, self._p.topics.goal, self._goal_cb, 10)

        if self._p.camera.publish_camera_info:
            CameraInfoPublisher(
                self,
                self._p.topics.rgb_info,
                self._p.topics.depth_info,
                self._p.camera.width,
                self._p.camera.height,
                self._p.camera.hfov_deg,
                self._p.frames.camera,
                self._p.camera.camera_info_rate_hz,
            )

        self._depth_img: Optional[np.ndarray] = None
        self._depth_stamp: Optional[Time] = None
        self._depth_encoding: Optional[str] = None

        self._costmap_builder: Optional[DepthCostmap] = None
        self._last_costmap: Optional[CostmapResult] = None

        self._goal_map: Optional[Tuple[float, float]] = None
        self._goal_time: Optional[Time] = None
        self._path_base: List[Tuple[float, float]] = []
        self._last_plan_goal: Optional[Tuple[float, float]] = None

        self._progress_pose: Optional[Tuple[float, float]] = None
        self._progress_time: Optional[Time] = None
        self._last_auto_goal_time: Optional[Time] = None

        self._plan_timer = self.create_timer(1.0 / max(0.1, self._p.navigator.plan_hz), self._plan_cycle)
        self._control_timer = self.create_timer(
            1.0 / max(0.1, self._p.navigator.control_hz), self._control_cycle
        )

        self._publish_status("IDLE")

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _rgb_cb(self, msg: Image) -> None:
        pass

    def _depth_cb(self, msg: Image) -> None:
        encoding = msg.encoding or self._p.camera.depth_encoding
        if encoding not in ("32FC1", "16UC1"):
            if self._depth_encoding != encoding:
                self.get_logger().warn(f"Unsupported depth encoding: {encoding}")
            return

        dtype = np.float32 if encoding == "32FC1" else np.uint16
        depth = np.frombuffer(msg.data, dtype=dtype)
        if depth.size != msg.width * msg.height:
            return
        depth = depth.reshape((msg.height, msg.width))

        self._depth_img = depth
        self._depth_stamp = Time.from_msg(msg.header.stamp)
        self._depth_encoding = encoding
        if self._p.camera.depth_encoding != encoding:
            self._p.camera.depth_encoding = encoding

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._p.camera.fx = float(msg.k[0])
        self._p.camera.fy = float(msg.k[4])
        self._p.camera.cx = float(msg.k[2])
        self._p.camera.cy = float(msg.k[5])
        self._p.camera.width = int(msg.width)
        self._p.camera.height = int(msg.height)
        self._ensure_costmap_builder()

    def _goal_cb(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id
        if frame and frame != self._p.frames.map:
            self.get_logger().warn(f"Goal frame {frame} != {self._p.frames.map}. Ignoring.")
            return
        self._goal_map = (float(msg.pose.position.x), float(msg.pose.position.y))
        self._goal_time = Time.from_msg(msg.header.stamp)
        self._path_base = []
        self._last_plan_goal = None
        self._progress_pose = None
        self._progress_time = None
        self._publish_status("GOAL_RECEIVED")

    def _ensure_intrinsics(self) -> None:
        if self._p.camera.fx > 0.0 and self._p.camera.fy > 0.0:
            return
        width = float(self._p.camera.width)
        hfov_rad = math.radians(float(self._p.camera.hfov_deg))
        fx = 0.5 * width / math.tan(0.5 * hfov_rad)
        fy = fx
        self._p.camera.fx = fx
        self._p.camera.fy = fy
        self._p.camera.cx = float(self._p.camera.width) * 0.5
        self._p.camera.cy = float(self._p.camera.height) * 0.5

    def _ensure_costmap_builder(self) -> None:
        self._ensure_intrinsics()
        if self._p.camera.fx <= 0.0:
            return
        self._costmap_builder = DepthCostmap(self._p.costmap, self._p.camera)

    def _get_depth_age(self, now: Time) -> float:
        if self._depth_stamp is None:
            return 1e9
        return (now - self._depth_stamp).nanoseconds * 1e-9

    def _maybe_set_auto_goal(self, pose_xyyaw: Tuple[float, float, float]) -> None:
        if not self._p.navigator.auto_goal_enable:
            return
        now = self.get_clock().now()
        if self._goal_map is not None:
            return
        if self._last_auto_goal_time is not None:
            elapsed = (now - self._last_auto_goal_time).nanoseconds * 1e-9
            if elapsed < float(self._p.navigator.auto_goal_interval):
                return
        x, y, yaw = pose_xyyaw
        dist = float(self._p.navigator.auto_goal_distance)
        gx = x + math.cos(yaw) * dist
        gy = y + math.sin(yaw) * dist
        self._goal_map = (gx, gy)
        self._goal_time = now
        self._last_auto_goal_time = now
        self._publish_status("AUTO_GOAL_SET")

    def _plan_cycle(self) -> None:
        now = self.get_clock().now()
        pose_status = self._tf.get_pose_map_base()
        tracking_ok = pose_status.status == TRACKING_OK and pose_status.pose_map_base is not None
        if not tracking_ok:
            self._publish_status("TRACKING_LOST")
            return

        self._maybe_set_auto_goal(pose_status.pose_map_base)

        if self._goal_map is None:
            self._publish_status("IDLE")
            return

        if not self._p.navigator.replan_on_costmap and self._path_base and self._last_plan_goal == self._goal_map:
            return

        if self._last_costmap is None:
            self._publish_status("NO_COSTMAP")
            return

        goal_bl = self._goal_in_base(pose_status.pose_map_base, self._goal_map)
        start = (0.0, 0.0)
        path_msg = self._planner.plan(self._last_costmap.grid, start, goal_bl)
        if not path_msg.poses:
            self._path_base = []
            self._publish_status("NO_PATH")
            return

        self._path_base = [(p.pose.position.x, p.pose.position.y) for p in path_msg.poses]
        self._last_plan_goal = self._goal_map
        map_path = self._path_to_map(path_msg, pose_status.pose_map_base)
        map_path.header.stamp = now.to_msg()
        self._path_pub.publish(map_path)
        self._publish_status("PLANNED")

    def _goal_in_base(self, pose_map: Tuple[float, float, float], goal_map: Tuple[float, float]) -> Tuple[float, float]:
        px, py, yaw = pose_map
        dx = goal_map[0] - px
        dy = goal_map[1] - py
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        gx = dx * cos_y - dy * sin_y
        gy = dx * sin_y + dy * cos_y
        return gx, gy

    def _path_to_map(self, path_bl: Path, pose_map: Tuple[float, float, float]) -> Path:
        px, py, yaw = pose_map
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        out = Path()
        out.header.frame_id = self._p.frames.map
        for pose in path_bl.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            mx = px + cos_y * x - sin_y * y
            my = py + sin_y * x + cos_y * y
            new_pose = PoseStamped()
            new_pose.header.frame_id = self._p.frames.map
            new_pose.pose.position.x = float(mx)
            new_pose.pose.position.y = float(my)
            new_pose.pose.position.z = 0.0
            out.poses.append(new_pose)
        return out

    def _control_cycle(self) -> None:
        now = self.get_clock().now()
        pose_status = self._tf.get_pose_map_base()
        tracking_ok = pose_status.status == TRACKING_OK and pose_status.pose_map_base is not None

        if self._costmap_builder is None:
            self._ensure_costmap_builder()

        depth_age = self._get_depth_age(now)
        if self._depth_img is not None and depth_age <= float(self._p.navigator.max_depth_age):
            header = None
            if self._depth_stamp is not None:
                header = Header()
                header.stamp = self._depth_stamp.to_msg()
                header.frame_id = self._p.frames.base_link
            if self._costmap_builder is not None:
                self._last_costmap = self._costmap_builder.build_costmap(self._depth_img, header)
                self._costmap_pub.publish(self._last_costmap.grid)
        else:
            if self._depth_img is None:
                self._publish_status("WAITING_DEPTH")
            else:
                self._publish_status("DEPTH_STALE")

        collision = False
        if self._costmap_builder is not None and self._depth_img is not None:
            collision = self._costmap_builder.check_collision_ahead(self._depth_img)

        stuck = self._check_stuck(pose_status.pose_map_base if tracking_ok else None, now)

        in_recovery, rec_cmd = self._recovery.update(now, tracking_ok, stuck)
        if in_recovery:
            self._cmd_pub.publish(rec_cmd)
            self._publish_status("RECOVERY")
            return

        cmd, goal_reached = self._controller.compute_cmd(now, self._path_base, collision)
        if collision:
            self._publish_status("COLLISION_STOP")
        elif goal_reached:
            self._publish_status("GOAL_REACHED")
            self._goal_map = None
            self._path_base = []
            self._last_plan_goal = None
            self._controller.reset()
        else:
            if self._path_base:
                self._publish_status("FOLLOWING")
            else:
                self._publish_status("IDLE")

        self._cmd_pub.publish(cmd)

    def _check_stuck(self, pose_map: Optional[Tuple[float, float, float]], now: Time) -> bool:
        if pose_map is None or not self._path_base:
            self._progress_pose = None
            self._progress_time = None
            return False

        px, py, _ = pose_map
        if self._progress_pose is None or self._progress_time is None:
            self._progress_pose = (px, py)
            self._progress_time = now
            return False

        dist = math.hypot(px - self._progress_pose[0], py - self._progress_pose[1])
        if dist >= float(self._p.recovery.min_progress):
            self._progress_pose = (px, py)
            self._progress_time = now
            return False

        elapsed = (now - self._progress_time).nanoseconds * 1e-9
        return elapsed >= float(self._p.recovery.stuck_time)


def main() -> None:
    rclpy.init()
    node = NavigatorNode()
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
