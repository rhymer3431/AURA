from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("autonomy_stack")
    nav2_share = get_package_share_directory("nav2_bringup")

    default_nav2_params = os.path.join(pkg_share, "config", "nav2_orbslam.yaml")
    default_frontier_params = os.path.join(pkg_share, "config", "frontier_explorer.yaml")
    navigation_launch = os.path.join(nav2_share, "launch", "navigation_launch.py")

    use_sim_time = LaunchConfiguration("use_sim_time")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    frontier_params_file = LaunchConfiguration("frontier_params_file")
    auto_explore = LaunchConfiguration("auto_explore")
    pose_topic = LaunchConfiguration("pose_topic")
    map_points_topic = LaunchConfiguration("map_points_topic")
    map_topic = LaunchConfiguration("map_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    enable_semantic_octomap = LaunchConfiguration("enable_semantic_octomap")
    enable_octomap_server = LaunchConfiguration("enable_octomap_server")
    semantic_topic = LaunchConfiguration("semantic_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    rgb_info_topic = LaunchConfiguration("rgb_info_topic")
    semantic_marker_topic = LaunchConfiguration("semantic_marker_topic")
    semantic_octomap_cloud_topic = LaunchConfiguration("semantic_octomap_cloud_topic")
    semantic_projected_map_topic = LaunchConfiguration("semantic_projected_map_topic")
    octomap_resolution = LaunchConfiguration("octomap_resolution")
    octomap_frame_id = LaunchConfiguration("octomap_frame_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("nav2_params_file", default_value=default_nav2_params),
            DeclareLaunchArgument("frontier_params_file", default_value=default_frontier_params),
            DeclareLaunchArgument("auto_explore", default_value="true"),
            DeclareLaunchArgument("pose_topic", default_value="/orbslam/pose"),
            DeclareLaunchArgument("map_points_topic", default_value="/orbslam/map_points"),
            DeclareLaunchArgument("map_topic", default_value="/map"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("enable_semantic_octomap", default_value="false"),
            DeclareLaunchArgument("enable_octomap_server", default_value="false"),
            DeclareLaunchArgument("semantic_topic", default_value="/semantic/label"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth"),
            DeclareLaunchArgument("rgb_info_topic", default_value="/camera/rgb/camera_info"),
            DeclareLaunchArgument("semantic_marker_topic", default_value="/semantic_map/markers"),
            DeclareLaunchArgument("semantic_octomap_cloud_topic", default_value="/semantic_map/octomap_cloud"),
            DeclareLaunchArgument("semantic_projected_map_topic", default_value="/semantic_map/projected_map"),
            DeclareLaunchArgument("octomap_resolution", default_value="0.15"),
            DeclareLaunchArgument("octomap_frame_id", default_value="map"),
            Node(
                package="autonomy_stack",
                executable="pose_tf_bridge_node",
                name="pose_tf_bridge_node",
                output="screen",
                parameters=[
                    {
                        "pose_topic": pose_topic,
                        "odom_topic": odom_topic,
                        "map_frame": "map",
                        "odom_frame": "odom",
                        "base_frame": "base_link",
                        "publish_map_to_odom": True,
                        "flatten_to_2d": True,
                    }
                ],
            ),
            Node(
                package="autonomy_stack",
                executable="sparse_map_occupancy_node",
                name="sparse_map_occupancy_node",
                output="screen",
                parameters=[
                    {
                        "map_points_topic": map_points_topic,
                        "pose_topic": pose_topic,
                        "map_topic": map_topic,
                        "map_frame": "map",
                    }
                ],
            ),
            Node(
                package="autonomy_stack",
                executable="semantic_fusion_node",
                name="semantic_fusion_node",
                output="screen",
                condition=IfCondition(enable_semantic_octomap),
                parameters=[
                    {
                        "semantic_topic": semantic_topic,
                        "depth_topic": depth_topic,
                        "rgb_info_topic": rgb_info_topic,
                        "pose_topic": pose_topic,
                        "marker_topic": semantic_marker_topic,
                        "octomap_cloud_topic": semantic_octomap_cloud_topic,
                        "projected_map_topic": semantic_projected_map_topic,
                    }
                ],
            ),
            Node(
                package="octomap_server",
                executable="octomap_server_node",
                name="semantic_octomap_server",
                output="screen",
                condition=IfCondition(enable_octomap_server),
                parameters=[
                    {
                        "resolution": octomap_resolution,
                        "frame_id": octomap_frame_id,
                    }
                ],
                remappings=[
                    ("/cloud_in", semantic_octomap_cloud_topic),
                ],
            ),
            Node(
                package="autonomy_stack",
                executable="frontier_explorer_node",
                name="frontier_explorer_node",
                output="screen",
                condition=IfCondition(auto_explore),
                parameters=[frontier_params_file],
                remappings=[
                    ("/map", map_topic),
                    ("/orbslam/pose", pose_topic),
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(navigation_launch),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params_file,
                }.items(),
            ),
        ]
    )
