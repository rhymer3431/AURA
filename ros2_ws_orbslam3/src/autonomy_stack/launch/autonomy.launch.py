from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("autonomy_stack")
    default_params = os.path.join(pkg_share, "config", "autonomy.yaml")

    params_file = LaunchConfiguration("params_file")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    rgb_info_topic = LaunchConfiguration("rgb_info_topic")
    depth_info_topic = LaunchConfiguration("depth_info_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("rgb_topic", default_value="/habitat/rgb"),
            DeclareLaunchArgument("depth_topic", default_value="/habitat/depth"),
            DeclareLaunchArgument("rgb_info_topic", default_value="/habitat/rgb/camera_info"),
            DeclareLaunchArgument("depth_info_topic", default_value="/habitat/depth/camera_info"),
            Node(
                package="autonomy_stack",
                executable="navigator_node",
                name="navigator_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "topics.rgb": rgb_topic,
                        "topics.depth": depth_topic,
                        "topics.rgb_info": rgb_info_topic,
                        "topics.depth_info": depth_info_topic,
                    },
                ],
            ),
        ]
    )
