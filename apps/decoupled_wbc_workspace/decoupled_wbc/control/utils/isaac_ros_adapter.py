import threading
from dataclasses import dataclass

import msgpack
import msgpack_numpy as mnp
import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import ByteMultiArray
from tf2_msgs.msg import TFMessage

from decoupled_wbc.control.main.constants import (
    ISAAC_CLOCK_TOPIC,
    ISAAC_COMMAND_TOPIC,
    ISAAC_IMU_TOPIC,
    ISAAC_INTERNAL_COMMAND_TOPIC,
    ISAAC_INTERNAL_STATE_TOPIC,
    ISAAC_JOINT_STATES_TOPIC,
    ISAAC_TF_TOPIC,
)

# 29 DoF body order used across decoupled_wbc and gear_sonic_deploy.
G1_BODY_JOINT_ORDER_29 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_LEFT_HAND_JOINT_ORDER_7 = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]

G1_RIGHT_HAND_JOINT_ORDER_7 = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


def _byte_multi_array_to_bytes(data_field) -> bytes:
    raw = bytearray()
    for item in data_field:
        if isinstance(item, int):
            raw.append(item & 0xFF)
        elif isinstance(item, (bytes, bytearray)):
            raw.extend(item)
        else:
            raw.extend(bytes(item))
    return bytes(raw)


def _dict_to_byte_multi_array(payload: dict) -> ByteMultiArray:
    packed = msgpack.packb(payload, default=mnp.encode)
    msg = ByteMultiArray()
    # Keep compatibility with decoupled_wbc ROSMsgSubscriber flatten logic.
    msg.data = tuple(bytes([b]) for b in packed)
    return msg


def _byte_multi_array_to_dict(msg: ByteMultiArray) -> dict:
    return msgpack.unpackb(_byte_multi_array_to_bytes(msg.data), object_hook=mnp.decode)


def _fit_size(values, size: int, default: float = 0.0) -> np.ndarray:
    arr = np.full(size, default, dtype=np.float64)
    src = np.asarray(values, dtype=np.float64).reshape(-1)
    usable = min(size, src.size)
    if usable > 0:
        arr[:usable] = src[:usable]
    return arr


def _ordered_values(name_to_value: dict[str, float], ordered_names: list[str]) -> np.ndarray:
    return np.asarray([name_to_value.get(name, 0.0) for name in ordered_names], dtype=np.float64)


def _quat_xyzw_to_wxyz(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.asarray([w, x, y, z], dtype=np.float64)


@dataclass
class IsaacAdapterConfig:
    run_mode: str = "both"
    """one of: both, state, command"""

    with_hands: bool = True
    """when False, only 29 body joints are published on command topic"""

    state_publish_hz: float = 100.0
    """publishing rate for internal state topic"""

    joint_states_topic: str = ISAAC_JOINT_STATES_TOPIC
    imu_topic: str = ISAAC_IMU_TOPIC
    tf_topic: str = ISAAC_TF_TOPIC
    clock_topic: str = ISAAC_CLOCK_TOPIC
    base_frame: str = "base_link"
    odom_frame: str = "odom"

    internal_state_topic: str = ISAAC_INTERNAL_STATE_TOPIC
    internal_command_topic: str = ISAAC_INTERNAL_COMMAND_TOPIC
    isaac_command_topic: str = ISAAC_COMMAND_TOPIC


class IsaacToInternalStateBridge(Node):
    def __init__(self, config: IsaacAdapterConfig):
        super().__init__("isaac_to_internal_state_bridge")
        self._config = config
        self._lock = threading.Lock()

        self._joint_pos: dict[str, float] = {}
        self._joint_vel: dict[str, float] = {}
        self._joint_eff: dict[str, float] = {}
        self._imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._imu_ang_vel = np.zeros(3, dtype=np.float64)
        self._imu_lin_acc = np.zeros(3, dtype=np.float64)
        self._base_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._clock_sec = 0.0

        self.create_subscription(JointState, config.joint_states_topic, self._joint_states_cb, 50)
        self.create_subscription(Imu, config.imu_topic, self._imu_cb, 50)
        self.create_subscription(TFMessage, config.tf_topic, self._tf_cb, 50)
        self.create_subscription(Clock, config.clock_topic, self._clock_cb, 50)
        self._publisher = self.create_publisher(ByteMultiArray, config.internal_state_topic, 50)

        self._timer = self.create_timer(1.0 / config.state_publish_hz, self._publish_state)

    def _joint_states_cb(self, msg: JointState):
        with self._lock:
            self._joint_pos = {
                name: msg.position[i] if i < len(msg.position) else 0.0
                for i, name in enumerate(msg.name)
            }
            self._joint_vel = {
                name: msg.velocity[i] if i < len(msg.velocity) else 0.0
                for i, name in enumerate(msg.name)
            }
            self._joint_eff = {
                name: msg.effort[i] if i < len(msg.effort) else 0.0 for i, name in enumerate(msg.name)
            }

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._imu_quat_wxyz = _quat_xyzw_to_wxyz(
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
            )
            self._imu_ang_vel = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                dtype=np.float64,
            )
            self._imu_lin_acc = np.array(
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                dtype=np.float64,
            )

    def _tf_cb(self, msg: TFMessage):
        selected = None
        for tf in msg.transforms:
            if tf.child_frame_id != self._config.base_frame:
                continue
            if tf.header.frame_id == self._config.odom_frame:
                selected = tf
                break
            if selected is None:
                selected = tf

        if selected is None:
            return

        with self._lock:
            quat_wxyz = _quat_xyzw_to_wxyz(
                selected.transform.rotation.x,
                selected.transform.rotation.y,
                selected.transform.rotation.z,
                selected.transform.rotation.w,
            )
            self._base_pose = np.array(
                [
                    selected.transform.translation.x,
                    selected.transform.translation.y,
                    selected.transform.translation.z,
                    quat_wxyz[0],
                    quat_wxyz[1],
                    quat_wxyz[2],
                    quat_wxyz[3],
                ],
                dtype=np.float64,
            )

    def _clock_cb(self, msg: Clock):
        with self._lock:
            self._clock_sec = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9

    def _publish_state(self):
        with self._lock:
            body_q = _ordered_values(self._joint_pos, G1_BODY_JOINT_ORDER_29)
            body_dq = _ordered_values(self._joint_vel, G1_BODY_JOINT_ORDER_29)
            body_tau_est = _ordered_values(self._joint_eff, G1_BODY_JOINT_ORDER_29)

            left_hand_q = _ordered_values(self._joint_pos, G1_LEFT_HAND_JOINT_ORDER_7)
            right_hand_q = _ordered_values(self._joint_pos, G1_RIGHT_HAND_JOINT_ORDER_7)
            left_hand_dq = _ordered_values(self._joint_vel, G1_LEFT_HAND_JOINT_ORDER_7)
            right_hand_dq = _ordered_values(self._joint_vel, G1_RIGHT_HAND_JOINT_ORDER_7)
            left_hand_tau_est = _ordered_values(self._joint_eff, G1_LEFT_HAND_JOINT_ORDER_7)
            right_hand_tau_est = _ordered_values(self._joint_eff, G1_RIGHT_HAND_JOINT_ORDER_7)

            ros_timestamp = self._clock_sec if self._clock_sec > 0.0 else self.get_clock().now().nanoseconds / 1e9

            payload = {
                "floating_base_pose": self._base_pose.tolist(),
                "floating_base_vel": np.concatenate([np.zeros(3), self._imu_ang_vel]).tolist(),
                "floating_base_acc": np.concatenate([self._imu_lin_acc, np.zeros(3)]).tolist(),
                "body_q": body_q.tolist(),
                "body_dq": body_dq.tolist(),
                "body_ddq": np.zeros(29, dtype=np.float64).tolist(),
                "body_tau_est": body_tau_est.tolist(),
                "left_hand_q": left_hand_q.tolist(),
                "left_hand_dq": left_hand_dq.tolist(),
                "left_hand_ddq": np.zeros(7, dtype=np.float64).tolist(),
                "left_hand_tau_est": left_hand_tau_est.tolist(),
                "right_hand_q": right_hand_q.tolist(),
                "right_hand_dq": right_hand_dq.tolist(),
                "right_hand_ddq": np.zeros(7, dtype=np.float64).tolist(),
                "right_hand_tau_est": right_hand_tau_est.tolist(),
                "torso_quat": self._imu_quat_wxyz.tolist(),
                "torso_ang_vel": self._imu_ang_vel.tolist(),
                "ros_timestamp": ros_timestamp,
                "foot_contact": [],
            }

        self._publisher.publish(_dict_to_byte_multi_array(payload))


class InternalCommandToIsaacBridge(Node):
    def __init__(self, config: IsaacAdapterConfig):
        super().__init__("internal_command_to_isaac_bridge")
        self._config = config
        self._lock = threading.Lock()

        self._body_q = np.zeros(29, dtype=np.float64)
        self._body_dq = np.zeros(29, dtype=np.float64)
        self._body_tau = np.zeros(29, dtype=np.float64)
        self._left_hand_q = np.zeros(7, dtype=np.float64)
        self._right_hand_q = np.zeros(7, dtype=np.float64)
        self._clock_msg: Clock | None = None

        self.create_subscription(ByteMultiArray, config.internal_command_topic, self._command_cb, 50)
        self.create_subscription(Clock, config.clock_topic, self._clock_cb, 50)
        self._publisher = self.create_publisher(JointState, config.isaac_command_topic, 50)

    def _clock_cb(self, msg: Clock):
        with self._lock:
            self._clock_msg = msg

    def _command_cb(self, msg: ByteMultiArray):
        try:
            payload = _byte_multi_array_to_dict(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode internal command payload: {exc}")
            return

        with self._lock:
            if "body_q" in payload:
                self._body_q = _fit_size(payload["body_q"], 29)
            if "body_dq" in payload:
                self._body_dq = _fit_size(payload["body_dq"], 29)
            if "body_tau" in payload:
                self._body_tau = _fit_size(payload["body_tau"], 29)
            if "left_hand_q" in payload:
                self._left_hand_q = _fit_size(payload["left_hand_q"], 7)
            if "right_hand_q" in payload:
                self._right_hand_q = _fit_size(payload["right_hand_q"], 7)

            cmd = JointState()
            if self._clock_msg is not None:
                cmd.header.stamp = self._clock_msg.clock
            else:
                cmd.header.stamp = self.get_clock().now().to_msg()

            if self._config.with_hands:
                cmd.name = (
                    G1_BODY_JOINT_ORDER_29 + G1_LEFT_HAND_JOINT_ORDER_7 + G1_RIGHT_HAND_JOINT_ORDER_7
                )
                cmd.position = np.concatenate(
                    [self._body_q, self._left_hand_q, self._right_hand_q], axis=0
                ).tolist()
                cmd.velocity = np.concatenate(
                    [self._body_dq, np.zeros(14, dtype=np.float64)], axis=0
                ).tolist()
                cmd.effort = np.concatenate(
                    [self._body_tau, np.zeros(14, dtype=np.float64)], axis=0
                ).tolist()
            else:
                cmd.name = G1_BODY_JOINT_ORDER_29
                cmd.position = self._body_q.tolist()
                cmd.velocity = self._body_dq.tolist()
                cmd.effort = self._body_tau.tolist()

        self._publisher.publish(cmd)
