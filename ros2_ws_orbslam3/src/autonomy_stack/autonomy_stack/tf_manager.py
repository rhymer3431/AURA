from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, StaticTransformBroadcaster

from .params import TFParams, Frames, TransformParams


TRACKING_OK = "TRACKING_OK"
TRACKING_LOST = "LOST"


@dataclass
class PoseStatus:
    status: str
    stamp: rclpy.time.Time
    pose_map_base: Optional[Tuple[float, float, float]]


def _rpy_to_quat(roll: float, pitch: float, yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def _transform_from_params(parent: str, child: str, params: TransformParams) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(params.x)
    tf.transform.translation.y = float(params.y)
    tf.transform.translation.z = float(params.z)
    qx, qy, qz, qw = _rpy_to_quat(params.roll, params.pitch, params.yaw)
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf


def _pose_to_xy_yaw(pose: PoseStamped) -> Tuple[float, float, float]:
    x = pose.pose.position.x
    y = pose.pose.position.y
    q = pose.pose.orientation
    # yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(x), float(y), float(yaw)


class TFManager:
    def __init__(self, node: rclpy.node.Node, tf_params: TFParams, frames: Frames) -> None:
        self._node = node
        self._tf_params = tf_params
        self._frames = frames
        self._buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._listener = TransformListener(self._buffer, node)
        self._tf_pub = TransformBroadcaster(node)
        self._static_tf_pub = StaticTransformBroadcaster(node)

        self._latest_pose: Optional[PoseStamped] = None
        self._latest_pose_stamp: Optional[rclpy.time.Time] = None

        if tf_params.publish_static_tf:
            static_tf = _transform_from_params(frames.base_link, frames.camera, tf_params.camera_in_base)
            static_tf.header.stamp = node.get_clock().now().to_msg()
            self._static_tf_pub.sendTransform(static_tf)

        if tf_params.use_pose_topic:
            node.create_subscription(PoseStamped, tf_params.pose_topic, self._pose_cb, 10)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._latest_pose = msg
        self._latest_pose_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if self._tf_params.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = self._frames.map if self._tf_params.pose_in_map_frame else self._frames.odom
            tf.child_frame_id = self._tf_params.pose_child_frame
            tf.transform.translation.x = msg.pose.position.x
            tf.transform.translation.y = msg.pose.position.y
            tf.transform.translation.z = msg.pose.position.z
            tf.transform.rotation = msg.pose.orientation
            self._tf_pub.sendTransform(tf)

    def get_pose_map_base(self) -> PoseStatus:
        now = self._node.get_clock().now()
        if self._tf_params.use_pose_topic and self._latest_pose is not None:
            if self._latest_pose_stamp is None:
                return PoseStatus(TRACKING_LOST, now, None)
            if (now - self._latest_pose_stamp) > Duration(seconds=float(self._tf_params.pose_timeout)):
                return PoseStatus(TRACKING_LOST, now, None)
            pose = _pose_to_xy_yaw(self._latest_pose)
            return PoseStatus(TRACKING_OK, now, pose)

        # Try TF lookup
        try:
            tf = self._buffer.lookup_transform(
                self._frames.map,
                self._frames.base_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            return PoseStatus(TRACKING_LOST, now, None)

        t = tf.transform.translation
        q = tf.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return PoseStatus(TRACKING_OK, now, (float(t.x), float(t.y), float(yaw)))
