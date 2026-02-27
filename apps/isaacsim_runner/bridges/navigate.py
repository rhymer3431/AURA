from __future__ import annotations

import logging
import math
import time

from apps.isaacsim_runner.config.base import DEFAULT_G1_START_Z, NAV_CMD_DEADBAND
from apps.isaacsim_runner.stage.prims import get_translate_op


class NavigateCommandBridge:
    """Simple twist subscriber that applies base XY/yaw motion to the robot root."""

    def __init__(self, namespace: str, robot_root_prim_path: str, command_timeout_s: float = 0.6) -> None:
        self.namespace = namespace.strip("/")
        self.robot_root_prim_path = robot_root_prim_path
        self.command_timeout_s = float(command_timeout_s)

        self.available = False
        self._rclpy = None
        self._node = None
        self._sub = None
        self._last_cmd_ts = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._last_warn_ts = 0.0
        self._last_rx_log_ts = 0.0

    def start(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("Navigate command bridge disabled (missing ROS2 libs): %s", exc)
            return

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("g1_navigate_command_bridge")
        topic = f"/{self.namespace}/cmd/navigate" if self.namespace else "/cmd/navigate"
        self._sub = self._node.create_subscription(Twist, topic, self._on_twist, 20)
        self.available = True
        logging.info(
            "Navigate command bridge ready: topic=%s target_prim=%s",
            topic,
            self.robot_root_prim_path,
        )

    def _on_twist(self, msg) -> None:
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._wz = float(msg.angular.z)
        self._last_cmd_ts = time.time()
        if (self._last_cmd_ts - self._last_rx_log_ts) > 0.7:
            logging.info(
                "Navigate command rx: vx=%.3f vy=%.3f wz=%.3f",
                self._vx,
                self._vy,
                self._wz,
            )
            self._last_rx_log_ts = self._last_cmd_ts

    def spin_once(self) -> None:
        if self.available and self._node is not None and self._rclpy is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def apply(self, stage_obj, dt: float) -> None:
        if not self.available:
            return
        if (time.time() - self._last_cmd_ts) > self.command_timeout_s:
            return
        if dt <= 0.0:
            return
        # Avoid re-authoring transform for effectively zero commands;
        # otherwise physics settling/gravity can be unintentionally suppressed.
        if (abs(self._vx) + abs(self._vy) + abs(self._wz)) <= NAV_CMD_DEADBAND:
            return

        try:
            from pxr import Gf, UsdGeom  # type: ignore
        except Exception as exc:
            logging.warning("Could not import pxr modules for navigate command apply: %s", exc)
            return

        prim = stage_obj.GetPrimAtPath(self.robot_root_prim_path)
        if not prim.IsValid():
            if time.time() - self._last_warn_ts > 2.0:
                logging.warning("Navigate command target prim missing: %s", self.robot_root_prim_path)
                self._last_warn_ts = time.time()
            return

        xform = UsdGeom.Xformable(prim)
        translate_op = get_translate_op(xform)
        rotate_z_op = None
        for op in xform.GetOrderedXformOps():
            if rotate_z_op is None and op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_z_op = op

        if translate_op is None:
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, DEFAULT_G1_START_Z))
        if rotate_z_op is None:
            rotate_z_op = xform.AddRotateZOp()
            rotate_z_op.Set(0.0)

        pos = translate_op.Get() or Gf.Vec3d(0.0, 0.0, DEFAULT_G1_START_Z)
        yaw_deg = float(rotate_z_op.Get() or 0.0)
        yaw = math.radians(yaw_deg)
        world_vx = self._vx * math.cos(yaw) - self._vy * math.sin(yaw)
        world_vy = self._vx * math.sin(yaw) + self._vy * math.cos(yaw)

        next_pos = Gf.Vec3d(
            float(pos[0]) + float(world_vx * dt),
            float(pos[1]) + float(world_vy * dt),
            float(pos[2]),
        )
        next_yaw = float(yaw_deg + math.degrees(self._wz * dt))
        translate_op.Set(next_pos)
        rotate_z_op.Set(next_yaw)

    def stop(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._node = None
        self._sub = None
