import time

import numpy as np
import rclpy
import tyro

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC, DEFAULT_NAV_CMD, DEFAULT_WRIST_POSE
from decoupled_wbc.control.main.teleop.configs.configs import TeleopConfig
from decoupled_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher
from decoupled_wbc.control.utils.telemetry import Telemetry

TELEOP_NODE_NAME = "TeleopPolicy"


def _to_transform_4x4_list(value) -> list[list[float]]:
    if value is None:
        arr = np.eye(4, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(4, 4)
    return arr.tolist()


def _extract_hand_joint_7(finger_data) -> list[float]:
    if isinstance(finger_data, dict) and "angle" in finger_data:
        angles = np.asarray(finger_data["angle"], dtype=np.float64).reshape(-1)
        if angles.size >= 7:
            return angles[:7].tolist()
    return [0.0] * 7


def _ensure_ros2_control_goal_schema(data: dict, ros_timestamp: float):
    if "base_height_cmd" in data and "base_height_command" not in data:
        data["base_height_command"] = data["base_height_cmd"]

    data.setdefault("navigate_cmd", DEFAULT_NAV_CMD)
    data.setdefault("wrist_pose", DEFAULT_WRIST_POSE)

    left_wrist = data.get("left_wrist")
    right_wrist = data.get("right_wrist")
    data.setdefault("left_wrist_after_ik", _to_transform_4x4_list(left_wrist))
    data.setdefault("right_wrist_after_ik", _to_transform_4x4_list(right_wrist))
    data.setdefault("head_after_ik", np.eye(4, dtype=np.float64).tolist())

    data.setdefault("left_hand_joint", _extract_hand_joint_7(data.get("left_fingers")))
    data.setdefault("right_hand_joint", _extract_hand_joint_7(data.get("right_fingers")))

    data.setdefault("base_height_command", 0.74)
    data.setdefault("toggle_policy_action", False)
    data.setdefault("locomotion_mode", 0)
    data["ros_timestamp"] = ros_timestamp


def main(config: TeleopConfig):
    ros_manager = ROSManager(node_name=TELEOP_NODE_NAME)
    node = ros_manager.node

    if config.robot == "g1":
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
        )
        left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    if config.lerobot_replay_path:
        teleop_policy = LerobotReplayPolicy(
            robot_model=robot_model, parquet_path=config.lerobot_replay_path
        )
    else:
        print("running teleop policy, waiting teleop policy to be initialized...")
        retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=config.enable_visualization,
            body_active_joint_groups=["upper_body"],
        )
        teleop_policy = TeleopPolicy(
            robot_model=robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=config.body_control_device,
            hand_control_device=config.hand_control_device,
            body_streamer_ip=config.body_streamer_ip,  # vive tracker, leap motion does not require
            body_streamer_keyword=config.body_streamer_keyword,
            enable_real_device=config.enable_real_device,
            replay_data_path=config.teleop_replay_path,
        )

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    telemetry = Telemetry(window_size=100)

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()

                # Add timing information to the message
                t_now = time.monotonic()
                data["timestamp"] = t_now
                ros_now = node.get_clock().now().nanoseconds / 1e9

                # Set target completion time - longer for initial pose, then match control frequency
                if iteration == 0:
                    data["target_time"] = t_now + time_to_get_to_initial_pose
                else:
                    data["target_time"] = t_now + (1 / config.teleop_frequency)

                _ensure_ros2_control_goal_schema(data, ros_timestamp=ros_now)

                # Publish the teleop command
                with telemetry.timer("publish_teleop_command"):
                    control_publisher.publish(data)

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
            end_time = time.monotonic()
            if (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Teleop Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")

    finally:
        print("Cleaning up...")
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
