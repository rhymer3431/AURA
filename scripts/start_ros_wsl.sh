#!/usr/bin/env bash
set -euo pipefail

: "${ROS_DISTRO:=humble}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"

echo "[start_ros_wsl] ROS_DISTRO=${ROS_DISTRO}"
echo "[start_ros_wsl] Launching RTAB-Map + Nav2 placeholders."

# TODO: replace with your actual workspace and launch files.
# source ~/ros2_ws/install/setup.bash
# ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true
# ros2 launch rtabmap_launch rtabmap.launch.py \
#   rgb_topic:=/g1/camera/color/image_raw \
#   depth_topic:=/g1/camera/depth/image_raw \
#   frame_id:=g1/base_link \
#   approx_sync:=true
# ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true map:=/tmp/map.yaml

echo "[start_ros_wsl] TODO: fill launch commands for your RTAB-Map/Nav2 stack."

