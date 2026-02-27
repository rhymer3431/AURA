from __future__ import annotations

import argparse
import logging
import time

from apps.isaacsim_runner.config.base import DEFAULT_G1_START_Z


class MockRos2Publisher:
    """Optional ROS2 mock publisher for camera/depth/tf/joint_states/clock."""

    def __init__(self, namespace: str, publish_imu: bool, publish_compressed_color: bool) -> None:
        self.namespace = namespace.strip("/")
        self.publish_imu = publish_imu
        self.publish_compressed_color = publish_compressed_color
        self.available = False
        self.node = None
        self.executor = None
        self._imports = {}
        self._last_log = 0.0

        try:
            import rclpy
            from builtin_interfaces.msg import Time
            from geometry_msgs.msg import TransformStamped
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import CompressedImage, Image, Imu, JointState
            from tf2_msgs.msg import TFMessage

            self._imports = {
                "rclpy": rclpy,
                "Time": Time,
                "TransformStamped": TransformStamped,
                "SingleThreadedExecutor": SingleThreadedExecutor,
                "Node": Node,
                "Clock": Clock,
                "Image": Image,
                "CompressedImage": CompressedImage,
                "Imu": Imu,
                "JointState": JointState,
                "TFMessage": TFMessage,
            }
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("ROS2 python packages not found. Mock ROS2 publish disabled: %s", exc)

    def start(self) -> None:
        if not self.available:
            return
        rclpy = self._imports["rclpy"]
        Node = self._imports["Node"]
        Executor = self._imports["SingleThreadedExecutor"]
        Clock = self._imports["Clock"]
        Image = self._imports["Image"]
        JointState = self._imports["JointState"]
        TFMessage = self._imports["TFMessage"]
        TransformStamped = self._imports["TransformStamped"]
        Imu = self._imports["Imu"]
        CompressedImage = self._imports["CompressedImage"]

        rclpy.init(args=None)
        self.node = Node("isaac_runner_mock_pub")
        ns = f"/{self.namespace}" if self.namespace else ""

        self._color_pub = self.node.create_publisher(Image, f"{ns}/camera/color/image_raw", 10)
        self._depth_pub = self.node.create_publisher(Image, f"{ns}/camera/depth/image_raw", 10)
        self._joint_pub = self.node.create_publisher(JointState, f"{ns}/joint_states", 10)
        self._clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self._color_compressed_pub = None
        self._imu_pub = None
        if self.publish_compressed_color:
            self._color_compressed_pub = self.node.create_publisher(
                CompressedImage, f"{ns}/camera/color/image_raw/compressed", 10
            )
        if self.publish_imu:
            self._imu_pub = self.node.create_publisher(Imu, f"{ns}/imu", 10)

        self._msg_clock = Clock()
        self._msg_color = Image()
        self._msg_color.height = 480
        self._msg_color.width = 640
        self._msg_color.encoding = "rgb8"
        self._msg_color.step = self._msg_color.width * 3
        self._msg_color.data = bytes(self._msg_color.height * self._msg_color.step)
        self._msg_color.header.frame_id = f"{self.namespace}/camera_color_optical_frame"

        self._msg_depth = Image()
        self._msg_depth.height = 480
        self._msg_depth.width = 640
        self._msg_depth.encoding = "16UC1"
        self._msg_depth.step = self._msg_depth.width * 2
        self._msg_depth.data = bytes(self._msg_depth.height * self._msg_depth.step)
        self._msg_depth.header.frame_id = f"{self.namespace}/camera_depth_optical_frame"

        self._msg_joint = JointState()
        self._msg_joint.name = ["joint_1", "joint_2", "joint_3"]
        self._msg_joint.position = [0.0, 0.0, 0.0]

        self._msg_tf = TFMessage()
        tf = TransformStamped()
        tf.header.frame_id = "map"
        tf.child_frame_id = f"{self.namespace}/base_link"
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = DEFAULT_G1_START_Z
        tf.transform.rotation.w = 1.0
        self._msg_tf.transforms = [tf]

        self._msg_imu = Imu()
        self._msg_imu.header.frame_id = f"{self.namespace}/imu_link"
        self._msg_imu.orientation.w = 1.0
        self._msg_color_compressed = CompressedImage()
        self._msg_color_compressed.header.frame_id = f"{self.namespace}/camera_color_optical_frame"
        self._msg_color_compressed.format = "jpeg"
        self._msg_color_compressed.data = b""

        self.executor = Executor()
        self.executor.add_node(self.node)
        logging.info("Mock ROS2 publishers started on namespace '/%s'", self.namespace)

    def publish_once(self) -> None:
        if not self.available or self.node is None:
            return

        now = self.node.get_clock().now().to_msg()
        self._msg_color.header.stamp = now
        self._msg_depth.header.stamp = now
        self._msg_joint.header.stamp = now
        self._msg_clock.clock = now
        self._msg_tf.transforms[0].header.stamp = now
        self._msg_imu.header.stamp = now
        self._msg_color_compressed.header.stamp = now

        self._color_pub.publish(self._msg_color)
        self._depth_pub.publish(self._msg_depth)
        if self._color_compressed_pub is not None:
            self._color_compressed_pub.publish(self._msg_color_compressed)
        self._joint_pub.publish(self._msg_joint)
        self._clock_pub.publish(self._msg_clock)
        self._tf_pub.publish(self._msg_tf)
        if self._imu_pub is not None:
            self._imu_pub.publish(self._msg_imu)

        if time.time() - self._last_log >= 5.0:
            logging.info(
                "Publishing mock topics: /%s/camera/{color,depth}/image_raw, /%s/joint_states, /tf, /clock%s",
                self.namespace,
                self.namespace,
                (
                    f", /{self.namespace}/camera/color/image_raw/compressed"
                    if self.publish_compressed_color
                    else ""
                )
                + (f", /{self.namespace}/imu" if self.publish_imu else ""),
            )
            self._last_log = time.time()

    def spin_once(self) -> None:
        if self.executor is not None:
            self.executor.spin_once(timeout_sec=0.0)

    def stop(self) -> None:
        if not self.available:
            return
        rclpy = self._imports["rclpy"]
        if self.executor is not None and self.node is not None:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_mock_loop(args: argparse.Namespace) -> None:
    ros2 = MockRos2Publisher(args.namespace, args.publish_imu, args.publish_compressed_color)
    ros2.start()

    logging.info("Running Isaac Sim mock loop with USD: %s", args.usd)
    logging.info("USD transform edits are intentionally disabled.")

    try:
        sleep_s = max(0.01, 1.0 / float(args.rate_hz))
        last_heartbeat = 0.0
        while True:
            ros2.publish_once()
            ros2.spin_once()
            if not ros2.available and (time.time() - last_heartbeat) >= 5.0:
                logging.info(
                    "Mock heartbeat: would publish RGB/Depth/TF/JointState using D455 in g1_d455.usd."
                )
                last_heartbeat = time.time()
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("Mock runner interrupted by user.")
    finally:
        ros2.stop()
