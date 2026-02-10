from __future__ import annotations

import math
from typing import Callable, Optional

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CameraInfo, Image


def sensor_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )


def create_image_subscribers(
    node: rclpy.node.Node,
    rgb_topic: str,
    depth_topic: str,
    rgb_cb: Callable[[Image], None],
    depth_cb: Callable[[Image], None],
) -> None:
    qos = sensor_qos()
    node.create_subscription(Image, rgb_topic, rgb_cb, qos)
    node.create_subscription(Image, depth_topic, depth_cb, qos)


class CameraInfoPublisher:
    def __init__(
        self,
        node: rclpy.node.Node,
        rgb_info_topic: str,
        depth_info_topic: str,
        width: int,
        height: int,
        hfov_deg: float,
        frame_id: str,
        rate_hz: float = 1.0,
    ) -> None:
        self._node = node
        self._rgb_pub = node.create_publisher(CameraInfo, rgb_info_topic, 1)
        self._depth_pub = node.create_publisher(CameraInfo, depth_info_topic, 1)
        self._width = width
        self._height = height
        self._hfov_deg = hfov_deg
        self._frame_id = frame_id
        self._timer = node.create_timer(max(0.1, 1.0 / max(rate_hz, 0.1)), self._on_timer)

    def _make_camera_info(self) -> CameraInfo:
        hfov_rad = math.radians(self._hfov_deg)
        fx = 0.5 * self._width / math.tan(0.5 * hfov_rad)
        fy = fx
        cx = self._width * 0.5
        cy = self._height * 0.5
        info = CameraInfo()
        info.header.frame_id = self._frame_id
        info.width = self._width
        info.height = self._height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _on_timer(self) -> None:
        stamp = self._node.get_clock().now().to_msg()
        info = self._make_camera_info()
        info.header.stamp = stamp
        self._rgb_pub.publish(info)
        self._depth_pub.publish(info)


def safe_image_size(msg: Image) -> Optional[tuple[int, int]]:
    if msg is None:
        return None
    return int(msg.width), int(msg.height)
