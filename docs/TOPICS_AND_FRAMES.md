# Topics and Frames

## Isaac Sim publish namespace (`/g1`)

- `/g1/camera/color/image_raw`
- `/g1/camera/depth/image_raw`
- `/g1/imu` (optional)
- `/g1/joint_states`
- `/tf`
- `/tf_static` (recommended in real deployment)
- `/clock`

## RTAB-Map integration (expected)

- Inputs:
  - `/g1/camera/color/image_raw` (or remapped color topic)
  - `/g1/camera/depth/image_raw`
  - `/tf`
- Outputs:
  - `/rtabmap/localization_pose`
  - `/rtabmap/grid_map` or `/map`
  - `/rtabmap/info` (if enabled)

## Nav2 integration

- Action: `nav2_msgs/action/NavigateToPose`
- Agent adapter file: `apps/agent_runtime/modules/nav2_client.py`

## Agent internal contracts

- `Detection2D3D`
  - `{class_name, score, bbox/mask, position_in_map(optional), timestamp}`
- `ObjectMemoryEntry`
  - `{object_id, class_name, map_pose, last_seen, confidence, importance}`
- `Plan`
  - `list[SkillCall{name, args, success_criteria, retry_policy}]`

## G1 action adapter topics (optional ROS2 output)

- `/g1/cmd/left_arm` (`std_msgs/Float64MultiArray`)
- `/g1/cmd/right_arm` (`std_msgs/Float64MultiArray`)
- `/g1/cmd/left_hand` (`std_msgs/Float64MultiArray`)
- `/g1/cmd/right_hand` (`std_msgs/Float64MultiArray`)
- `/g1/cmd/waist` (`std_msgs/Float64MultiArray`)
- `/g1/cmd/joint_commands` (`sensor_msgs/JointState`)
- `/g1/cmd/base_height` (`std_msgs/Float64`)
- `/g1/cmd/navigate` (`geometry_msgs/Twist`)

## Frames

- Global map frame: `map`
- Robot base frame: `g1/base_link` (recommended)
- Camera frames:
  - `g1/camera_color_optical_frame`
  - `g1/camera_depth_optical_frame`

## Map-anchored object memory

- Scene memory stores object pose in `map` frame.
- If detector does not provide 3D map position directly, runtime anchors using current robot pose + heuristic projection.
- Real deployment should replace heuristic anchoring with depth+TF triangulation.
