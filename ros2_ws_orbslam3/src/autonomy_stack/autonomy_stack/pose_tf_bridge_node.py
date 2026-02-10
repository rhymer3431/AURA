from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class PoseTfBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_tf_bridge_node")

        self.declare_parameter("pose_topic", "/orbslam/pose")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_map_to_odom", True)
        self.declare_parameter("flatten_to_2d", True)
        self.declare_parameter("max_dt_for_twist", 0.3)

        pose_topic = str(self.get_parameter("pose_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._publish_map_to_odom = bool(self.get_parameter("publish_map_to_odom").value)
        self._flatten_to_2d = bool(self.get_parameter("flatten_to_2d").value)
        self._max_dt_for_twist = max(0.01, float(self.get_parameter("max_dt_for_twist").value))

        self._tf_pub = TransformBroadcaster(self)
        self._static_tf_pub = StaticTransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, odom_topic, 20)

        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        self._prev_t: Optional[float] = None

        if self._publish_map_to_odom:
            static_tf = TransformStamped()
            static_tf.header.stamp = self.get_clock().now().to_msg()
            static_tf.header.frame_id = self._map_frame
            static_tf.child_frame_id = self._odom_frame
            static_tf.transform.rotation.w = 1.0
            self._static_tf_pub.sendTransform(static_tf)

        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 20)
        self.get_logger().info(
            f"Pose bridge started: pose={pose_topic}, tf={self._map_frame}->{self._odom_frame}->{self._base_frame}, odom={odom_topic}"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        t_sec = _stamp_to_sec(stamp)

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)

        if self._flatten_to_2d:
            yaw = _quat_to_yaw(qx, qy, qz, qw)
            qx, qy, qz, qw = _yaw_to_quat(yaw)
            z = 0.0
        else:
            yaw = _quat_to_yaw(qx, qy, qz, qw)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._odom_frame
        tf.child_frame_id = self._base_frame
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf_pub.sendTransform(tf)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[14] = 0.2
        odom.pose.covariance[21] = 0.2
        odom.pose.covariance[28] = 0.2
        odom.pose.covariance[35] = 0.1

        vx = 0.0
        vy = 0.0
        wz = 0.0
        if (
            self._prev_x is not None
            and self._prev_y is not None
            and self._prev_yaw is not None
            and self._prev_t is not None
        ):
            dt = t_sec - self._prev_t
            if 1e-4 < dt <= self._max_dt_for_twist:
                vx = (x - self._prev_x) / dt
                vy = (y - self._prev_y) / dt
                wz = _wrap_to_pi(yaw - self._prev_yaw) / dt

        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.angular.z = float(wz)
        odom.twist.covariance[0] = 0.2
        odom.twist.covariance[7] = 0.2
        odom.twist.covariance[35] = 0.2
        self._odom_pub.publish(odom)

        self._prev_x = x
        self._prev_y = y
        self._prev_yaw = yaw
        self._prev_t = t_sec


def main() -> None:
    rclpy.init()
    node = PoseTfBridgeNode()
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
