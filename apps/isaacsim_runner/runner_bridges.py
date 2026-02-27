from __future__ import annotations

import argparse
import logging
import math
import time

from apps.isaacsim_runner.runner_config import DEFAULT_G1_START_Z, NAV_CMD_DEADBAND
from apps.isaacsim_runner.runner_stage import _get_translate_op


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
        except Exception:
            return

        prim = stage_obj.GetPrimAtPath(self.robot_root_prim_path)
        if not prim.IsValid():
            if time.time() - self._last_warn_ts > 2.0:
                logging.warning("Navigate command target prim missing: %s", self.robot_root_prim_path)
                self._last_warn_ts = time.time()
            return

        xform = UsdGeom.Xformable(prim)
        translate_op = _get_translate_op(xform)
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


def _run_mock_loop(args: argparse.Namespace) -> None:
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


def _setup_ros2_joint_and_tf_graph(namespace: str, robot_prim_path: str) -> None:
    import omni.graph.core as og  # type: ignore
    import usdrt.Sdf  # type: ignore

    graph_path = "/G1ROS2Bridge"
    cmd_topic = f"/{namespace}/cmd/joint_commands"
    joint_state_topic = f"/{namespace}/joint_states"

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "/clock"),
                ("PublishJointState.inputs:topicName", joint_state_topic),
                ("SubscribeJointState.inputs:topicName", cmd_topic),
                ("PublishTF.inputs:topicName", "/tf"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(robot_prim_path)]),
                ("ArticulationController.inputs:robotPath", robot_prim_path),
            ],
        },
    )

    logging.info(
        "ROS2 bridge graph ready: robot=%s, cmd_topic=%s, joint_state_topic=%s",
        robot_prim_path,
        cmd_topic,
        joint_state_topic,
    )


def _setup_ros2_camera_graph(namespace: str, camera_prim_path: str) -> None:
    import omni.graph.core as og  # type: ignore
    import usdrt.Sdf  # type: ignore

    graph_path = "/G1ROSCamera"
    color_topic = f"/{namespace}/camera/color/image_raw"
    depth_topic = f"/{namespace}/camera/depth/image_raw"
    camera_info_topic = f"/{namespace}/camera/color/camera_info"

    keys = og.Controller.Keys
    ros_camera_graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperDepth.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperRgb.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                # Keep GUI viewport interactive by using a dedicated offscreen render product.
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
                ("CreateRenderProduct.inputs:width", 640),
                ("CreateRenderProduct.inputs:height", 480),
                ("CameraHelperRgb.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperRgb.inputs:topicName", color_topic),
                ("CameraHelperRgb.inputs:type", "rgb"),
                ("CameraHelperInfo.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperInfo.inputs:topicName", camera_info_topic),
                ("CameraHelperDepth.inputs:frameId", f"{namespace}/camera_depth_optical_frame"),
                ("CameraHelperDepth.inputs:topicName", depth_topic),
                ("CameraHelperDepth.inputs:type", "depth"),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    logging.info(
        "ROS2 camera graph ready: camera=%s, topics=[%s, %s, %s]",
        camera_prim_path,
        color_topic,
        depth_topic,
        camera_info_topic,
    )
