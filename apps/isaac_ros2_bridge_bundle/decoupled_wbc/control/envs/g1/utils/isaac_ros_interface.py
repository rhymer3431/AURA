import threading
import time
from typing import Any, Dict, Optional

import msgpack
import msgpack_numpy as mnp
import numpy as np
from std_msgs.msg import ByteMultiArray

from decoupled_wbc.control.main.constants import (
    ISAAC_INTERNAL_COMMAND_TOPIC,
    ISAAC_INTERNAL_STATE_TOPIC,
)
from decoupled_wbc.control.utils.ros_utils import ROSManager


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


def _pack_dict_to_byte_multi_array(payload: dict) -> ByteMultiArray:
    packed = msgpack.packb(payload, default=mnp.encode)
    msg = ByteMultiArray()
    # Keep compatibility with existing ROS utils that flatten iterable bytes.
    msg.data = tuple(bytes([b]) for b in packed)
    return msg


def _decode_byte_multi_array(msg: ByteMultiArray) -> dict:
    packed = _byte_multi_array_to_bytes(msg.data)
    return msgpack.unpackb(packed, object_hook=mnp.decode)


def _safe_array(payload: Dict[str, Any], key: str, size: int, default: float = 0.0) -> np.ndarray:
    values = payload.get(key)
    arr = np.full(size, default, dtype=np.float64)
    if values is None:
        return arr
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    usable = min(size, values_arr.size)
    if usable > 0:
        arr[:usable] = values_arr[:usable]
    return arr


class _IsaacStateBuffer:
    _instances: Dict[str, "_IsaacStateBuffer"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def instance(cls, topic_name: str) -> "_IsaacStateBuffer":
        with cls._instances_lock:
            instance = cls._instances.get(topic_name)
            if instance is None:
                instance = cls(topic_name)
                cls._instances[topic_name] = instance
            return instance

    def __init__(self, topic_name: str):
        ros_manager = ROSManager(node_name="isaac_state_buffer")
        self._node = ros_manager.node
        self._lock = threading.Lock()
        self._latest_state: Optional[dict] = None
        self._subscription = self._node.create_subscription(
            ByteMultiArray, topic_name, self._callback, 10
        )

    def _callback(self, msg: ByteMultiArray):
        try:
            state = _decode_byte_multi_array(msg)
        except Exception as exc:
            self._node.get_logger().warning(f"Failed to decode Isaac state msgpack payload: {exc}")
            return
        with self._lock:
            self._latest_state = state

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            if self._latest_state is None:
                return None
            return dict(self._latest_state)


class _IsaacCommandPublisher:
    _instances: Dict[str, "_IsaacCommandPublisher"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def instance(cls, topic_name: str) -> "_IsaacCommandPublisher":
        with cls._instances_lock:
            instance = cls._instances.get(topic_name)
            if instance is None:
                instance = cls(topic_name)
                cls._instances[topic_name] = instance
            return instance

    def __init__(self, topic_name: str):
        ros_manager = ROSManager(node_name="isaac_command_publisher")
        self._node = ros_manager.node
        self._publisher = self._node.create_publisher(ByteMultiArray, topic_name, 10)
        self._lock = threading.Lock()

        self._body_q = np.zeros(29, dtype=np.float64)
        self._body_dq = np.zeros(29, dtype=np.float64)
        self._body_tau = np.zeros(29, dtype=np.float64)
        self._left_hand_q = np.zeros(7, dtype=np.float64)
        self._right_hand_q = np.zeros(7, dtype=np.float64)

    def update_body(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        with self._lock:
            self._body_q = np.asarray(cmd_q, dtype=np.float64).reshape(29)
            self._body_dq = np.asarray(cmd_dq, dtype=np.float64).reshape(29)
            self._body_tau = np.asarray(cmd_tau, dtype=np.float64).reshape(29)
            self._publish_locked(source="body")

    def update_hand(self, side: str, cmd_q: np.ndarray):
        hand_q = np.asarray(cmd_q, dtype=np.float64).reshape(7)
        with self._lock:
            if side == "left":
                self._left_hand_q = hand_q
            else:
                self._right_hand_q = hand_q
            self._publish_locked(source=f"{side}_hand")

    def _publish_locked(self, source: str):
        payload = {
            "body_q": self._body_q.tolist(),
            "body_dq": self._body_dq.tolist(),
            "body_tau": self._body_tau.tolist(),
            "left_hand_q": self._left_hand_q.tolist(),
            "right_hand_q": self._right_hand_q.tolist(),
            "ros_timestamp": time.time(),
            "source": source,
        }
        self._publisher.publish(_pack_dict_to_byte_multi_array(payload))


class IsaacBodyStateProcessor:
    def __init__(self, config: Dict[str, Any]):
        state_topic = config.get("ISAAC_INTERNAL_STATE_TOPIC", ISAAC_INTERNAL_STATE_TOPIC)
        self._state_buffer = _IsaacStateBuffer.instance(state_topic)
        self._last_robot_state = np.zeros((1, 148), dtype=np.float64)

    def _prepare_low_state(self) -> np.ndarray:
        state = self._state_buffer.get_latest()
        if state is None:
            return self._last_robot_state

        floating_base_pose = _safe_array(state, "floating_base_pose", 7)
        floating_base_vel = _safe_array(state, "floating_base_vel", 6)
        floating_base_acc = _safe_array(state, "floating_base_acc", 6)

        body_q = _safe_array(state, "body_q", 29)
        body_dq = _safe_array(state, "body_dq", 29)
        body_ddq = _safe_array(state, "body_ddq", 29)
        body_tau_est = _safe_array(state, "body_tau_est", 29)

        torso_quat = _safe_array(state, "torso_quat", 4)
        if np.linalg.norm(torso_quat) < 1e-9:
            torso_quat = floating_base_pose[3:7]

        torso_ang_vel = _safe_array(state, "torso_ang_vel", 3)
        if np.linalg.norm(torso_ang_vel) < 1e-9:
            torso_ang_vel = floating_base_vel[3:6]

        q = np.zeros(36, dtype=np.float64)
        dq = np.zeros(35, dtype=np.float64)
        ddq = np.zeros(35, dtype=np.float64)
        tau_est = np.zeros(35, dtype=np.float64)

        q[0:3] = floating_base_pose[0:3]
        q[3:7] = floating_base_pose[3:7]
        q[7:36] = body_q

        dq[0:3] = floating_base_vel[0:3]
        dq[3:6] = floating_base_vel[3:6]
        dq[6:35] = body_dq

        ddq[0:3] = floating_base_acc[0:3]
        ddq[3:6] = floating_base_acc[3:6]
        ddq[6:35] = body_ddq

        tau_est[6:35] = body_tau_est

        robot_state = np.concatenate([q, dq, tau_est, ddq, torso_quat, torso_ang_vel]).reshape(1, -1)
        if robot_state.shape != (1, 148):
            return self._last_robot_state

        self._last_robot_state = robot_state
        return robot_state


class IsaacHandStateProcessor:
    def __init__(self, config: Dict[str, Any], is_left: bool = True):
        self._is_left = is_left
        state_topic = config.get("ISAAC_INTERNAL_STATE_TOPIC", ISAAC_INTERNAL_STATE_TOPIC)
        self._state_buffer = _IsaacStateBuffer.instance(state_topic)
        self._last_hand_state = np.zeros((1, 28), dtype=np.float64)

    def _prepare_low_state(self) -> np.ndarray:
        state = self._state_buffer.get_latest()
        if state is None:
            return self._last_hand_state

        prefix = "left_hand" if self._is_left else "right_hand"
        hand_q = _safe_array(state, f"{prefix}_q", 7)
        hand_dq = _safe_array(state, f"{prefix}_dq", 7)
        hand_tau = _safe_array(state, f"{prefix}_tau_est", 7)
        hand_ddq = _safe_array(state, f"{prefix}_ddq", 7)

        hand_state = np.concatenate([hand_q, hand_dq, hand_tau, hand_ddq]).reshape(1, -1)
        if hand_state.shape != (1, 28):
            return self._last_hand_state

        self._last_hand_state = hand_state
        return hand_state


class IsaacBodyCommandSender:
    def __init__(self, config: Dict[str, Any]):
        command_topic = config.get("ISAAC_INTERNAL_COMMAND_TOPIC", ISAAC_INTERNAL_COMMAND_TOPIC)
        self._publisher = _IsaacCommandPublisher.instance(command_topic)

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        self._publisher.update_body(cmd_q, cmd_dq, cmd_tau)


class IsaacHandCommandSender:
    def __init__(self, config: Dict[str, Any], is_left: bool = True):
        command_topic = config.get("ISAAC_INTERNAL_COMMAND_TOPIC", ISAAC_INTERNAL_COMMAND_TOPIC)
        self._publisher = _IsaacCommandPublisher.instance(command_topic)
        self._side = "left" if is_left else "right"

    def send_command(self, cmd: np.ndarray):
        self._publisher.update_hand(self._side, cmd)
