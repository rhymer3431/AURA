from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_G1_GROUP_TO_JOINTS = {
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

_G1_BODY_29_JOINTS_FALLBACK = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


def _load_joint_names_from_map() -> List[str]:
    map_path = Path(__file__).resolve().parent / "g1_joint_map.json"
    try:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        joints = sorted(payload.get("joints", []), key=lambda item: int(item["sonic_idx"]))
        names = [str(item["name"]) for item in joints]
        if len(names) >= len(_G1_BODY_29_JOINTS_FALLBACK):
            return names[: len(_G1_BODY_29_JOINTS_FALLBACK)]
    except Exception as exc:
        logging.warning("Failed to load joint order from %s, using fallback list: %s", map_path, exc)
    return list(_G1_BODY_29_JOINTS_FALLBACK)


_G1_BODY_29_JOINTS = _load_joint_names_from_map()


class G1ActionAdapter:
    """Applies GR00T action outputs to a target control interface."""

    def __init__(self, cfg: Dict) -> None:
        self.backend = str(cfg.get("backend", "log")).strip().lower()
        self.topic_prefix = str(cfg.get("topic_prefix", "/g1/cmd"))
        self._lock = threading.RLock()

        self._rclpy = None
        self._node = None
        self._pubs = {}
        self._available = False
        self._last_joint_command: Optional[List[float]] = None
        self._last_joint_feedback: Optional[Dict[str, Any]] = None
        self._last_base_pose: Optional[Dict[str, float]] = None
        self._joint_state_sub = None
        self._tf_sub = None
        self._joint_state_topic = ""
        self._tf_topic = "/tf"

        if self.backend == "ros2_topic":
            self._init_ros2_publishers()
        else:
            self.backend = "log"
            self._available = True
            logging.warning(
                "G1 action adapter is in log mode. Robot motion commands are not published."
            )

    def _init_ros2_publishers(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Float64, Float64MultiArray

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node("g1_action_adapter")

            p = self.topic_prefix.rstrip("/")
            self._pubs = {
                "left_arm": self._node.create_publisher(Float64MultiArray, f"{p}/left_arm", 10),
                "right_arm": self._node.create_publisher(Float64MultiArray, f"{p}/right_arm", 10),
                "left_hand": self._node.create_publisher(Float64MultiArray, f"{p}/left_hand", 10),
                "right_hand": self._node.create_publisher(Float64MultiArray, f"{p}/right_hand", 10),
                "waist": self._node.create_publisher(Float64MultiArray, f"{p}/waist", 10),
                "base_height_command": self._node.create_publisher(Float64, f"{p}/base_height", 10),
                "navigate_command": self._node.create_publisher(Twist, f"{p}/navigate", 10),
                "base": self._node.create_publisher(Twist, f"{p}/navigate", 10),
                "joint_command": self._node.create_publisher(JointState, f"{p}/joint_commands", 10),
                "generic": self._node.create_publisher(Float64MultiArray, f"{p}/generic", 10),
            }
            state_topic = self._derive_joint_state_topic(p)
            self._joint_state_sub = self._node.create_subscription(
                JointState,
                state_topic,
                self._on_joint_state,
                50,
            )
            try:
                from tf2_msgs.msg import TFMessage

                self._tf_sub = self._node.create_subscription(
                    TFMessage,
                    self._tf_topic,
                    self._on_tf,
                    50,
                )
            except Exception as tf_exc:
                self._tf_sub = None
                logging.warning("TF subscription unavailable; base-pose feedback disabled: %s", tf_exc)
            self._joint_state_topic = state_topic
            self._available = True
            logging.info(
                "G1 action adapter initialized in ROS2 mode: prefix=%s joint_state_topic=%s tf_topic=%s",
                self.topic_prefix,
                self._joint_state_topic,
                self._tf_topic,
            )
        except Exception as exc:
            logging.warning("ROS2 action adapter unavailable, fallback to log mode: %s", exc)
            self.backend = "log"
            self._available = True

    def apply_action(self, action_step: Dict[str, List[float]], source: str = "groot") -> None:
        if not self._available:
            return
        with self._lock:
            if self.backend == "ros2_topic":
                self._publish_ros2(action_step, source)
            else:
                keys = sorted(action_step.keys())
                logging.info("G1 action[%s]: keys=%s", source, keys)

    def get_last_joint_command(self) -> Optional[List[float]]:
        with self._lock:
            if self._last_joint_command is None:
                return None
            return list(self._last_joint_command)

    def get_last_joint_feedback(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._last_joint_feedback is None:
                return None
            feedback = dict(self._last_joint_feedback)
            feedback["name"] = list(feedback.get("name", []))
            feedback["position"] = list(feedback.get("position", []))
            feedback["velocity"] = list(feedback.get("velocity", []))
            return feedback

    def get_last_base_pose(self) -> Optional[Dict[str, float]]:
        with self._lock:
            if self._last_base_pose is None:
                return None
            return dict(self._last_base_pose)

    @staticmethod
    def _derive_joint_state_topic(topic_prefix: str) -> str:
        p = topic_prefix.rstrip("/")
        if p.endswith("/cmd"):
            root = p[:-4]
        else:
            root = p
        root = root.strip()
        if not root:
            root = "/g1"
        if not root.startswith("/"):
            root = f"/{root}"
        return f"{root}/joint_states"

    @staticmethod
    def _quat_to_rpy(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _on_joint_state(self, msg) -> None:
        try:
            names = [str(v) for v in list(getattr(msg, "name", []))]
            positions = [float(v) for v in list(getattr(msg, "position", []))]
            velocities_raw = list(getattr(msg, "velocity", []))
            velocities = [float(v) for v in velocities_raw] if velocities_raw else [0.0] * len(positions)
            if len(velocities) < len(positions):
                velocities.extend([0.0] * (len(positions) - len(velocities)))
            stamp = float(time.time())
            if getattr(msg, "header", None) is not None and getattr(msg.header, "stamp", None) is not None:
                sec = float(getattr(msg.header.stamp, "sec", 0.0))
                nsec = float(getattr(msg.header.stamp, "nanosec", 0.0))
                if sec > 0.0 or nsec > 0.0:
                    stamp = sec + nsec * 1e-9
            payload: Dict[str, Any] = {
                "name": names,
                "position": positions,
                "velocity": velocities[: len(positions)],
                "timestamp": stamp,
                "received_timestamp": float(time.time()),
            }
            with self._lock:
                self._last_joint_feedback = payload
        except Exception as exc:
            logging.debug("Failed to parse joint state feedback: %s", exc)

    def _on_tf(self, msg) -> None:
        transforms = list(getattr(msg, "transforms", []))
        if not transforms:
            return
        preferred = "/g1_29dof_with_hand_rev_1_0/pelvis"
        target = None
        for tf in transforms:
            child = str(getattr(tf, "child_frame_id", ""))
            child_l = child.lower()
            if preferred in child_l:
                target = tf
                break
            if "pelvis" in child_l or "base_link" in child_l:
                target = tf
                break
        if target is None:
            target = transforms[0]
        try:
            tr = target.transform.translation
            rot = target.transform.rotation
            tx = float(getattr(tr, "x", 0.0))
            ty = float(getattr(tr, "y", 0.0))
            tz = float(getattr(tr, "z", 0.0))
            qx = float(getattr(rot, "x", 0.0))
            qy = float(getattr(rot, "y", 0.0))
            qz = float(getattr(rot, "z", 0.0))
            qw = float(getattr(rot, "w", 1.0))
            roll, pitch, yaw = self._quat_to_rpy(qw, qx, qy, qz)
            with self._lock:
                self._last_base_pose = {
                    "x": tx,
                    "y": ty,
                    "z": tz,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "timestamp": float(time.time()),
                }
        except Exception as exc:
            logging.debug("Failed to parse TF base pose: %s", exc)

    def _publish_ros2(self, action_step: Dict[str, List[float]], source: str) -> None:
        if self._node is None or self._rclpy is None:
            return

        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64, Float64MultiArray

        full_joint_values = None
        for key in ("joint_command", "joint_positions", "joint_actions"):
            values = action_step.get(key)
            if values:
                full_joint_values = [float(v) for v in values[: len(_G1_BODY_29_JOINTS)]]
                break

        if full_joint_values is not None:
            joint_cmd = JointState()
            joint_cmd.header.stamp = self._node.get_clock().now().to_msg()
            joint_cmd.name = list(_G1_BODY_29_JOINTS)
            joint_cmd.position = full_joint_values
            self._pubs["joint_command"].publish(joint_cmd)
            self._last_joint_command = list(full_joint_values)

        joint_names: List[str] = []
        joint_positions: List[float] = []
        for group, ordered_joints in _G1_GROUP_TO_JOINTS.items():
            values = action_step.get(group)
            if not values:
                continue
            for idx, joint_name in enumerate(ordered_joints):
                if idx >= len(values):
                    break
                joint_names.append(joint_name)
                joint_positions.append(float(values[idx]))
        if joint_names:
            joint_cmd = JointState()
            joint_cmd.header.stamp = self._node.get_clock().now().to_msg()
            joint_cmd.name = joint_names
            joint_cmd.position = joint_positions
            self._pubs["joint_command"].publish(joint_cmd)

        for key, value in action_step.items():
            if key in {"joint_command", "joint_positions", "joint_actions"}:
                continue
            if key in {"navigate_command", "base"}:
                twist = Twist()
                if len(value) > 0:
                    twist.linear.x = float(value[0])
                if len(value) > 1:
                    twist.linear.y = float(value[1])
                if len(value) > 2:
                    twist.angular.z = float(value[2])
                self._pubs["navigate_command"].publish(twist)
                continue

            if key == "base_height_command":
                msg = Float64()
                msg.data = float(value[0] if value else 0.0)
                self._pubs["base_height_command"].publish(msg)
                continue

            msg = Float64MultiArray()
            msg.data = [float(v) for v in value]
            pub = self._pubs.get(key, self._pubs["generic"])
            pub.publish(msg)

        # Keep outgoing comms active on some DDS stacks.
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        logging.debug("Published ROS2 G1 action from %s", source)

    def close(self) -> None:
        self._joint_state_sub = None
        self._tf_sub = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if self._rclpy is not None and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
