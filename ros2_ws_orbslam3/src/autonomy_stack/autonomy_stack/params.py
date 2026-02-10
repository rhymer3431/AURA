from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict
import copy
import yaml


@dataclass
class Topics:
    rgb: str = "/habitat/rgb"
    depth: str = "/habitat/depth"
    rgb_info: str = "/habitat/rgb/camera_info"
    depth_info: str = "/habitat/depth/camera_info"
    goal: str = "/goal_pose"
    cmd_vel: str = "/cmd_vel"
    costmap: str = "/costmap"
    global_path: str = "/global_path"
    nav_status: str = "/nav_status"


@dataclass
class Frames:
    map: str = "map"
    odom: str = "odom"
    base_link: str = "base_link"
    camera: str = "habitat_rgb"


@dataclass
class CameraParams:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 640
    height: int = 480
    hfov_deg: float = 90.0
    publish_camera_info: bool = False
    camera_info_rate_hz: float = 1.0
    depth_encoding: str = "32FC1"
    depth_scale: float = 1.0


@dataclass
class CostmapParams:
    resolution: float = 0.1
    size_x: float = 8.0
    size_y: float = 8.0
    min_depth: float = 0.2
    max_depth: float = 6.0
    min_height: float = 0.05
    max_height: float = 1.5
    camera_height: float = 0.5
    inflation_radius: float = 0.3
    pixel_stride: int = 2
    unknown_as_obstacle: bool = True
    safety_stop_distance: float = 0.6
    safety_fov_deg: float = 60.0


@dataclass
class PlannerParams:
    allow_diagonal: bool = True
    occupied_threshold: int = 50
    unknown_as_obstacle: bool = True
    max_iterations: int = 20000


@dataclass
class ControllerParams:
    lookahead_distance: float = 0.6
    linear_speed: float = 0.3
    max_linear_speed: float = 0.5
    max_angular_speed: float = 1.0
    max_linear_accel: float = 0.5
    max_angular_accel: float = 1.5
    goal_tolerance: float = 0.3
    yaw_tolerance: float = 0.4
    angular_kp: float = 1.5


@dataclass
class RecoveryParams:
    slam_lost_rotate_time: float = 3.0
    slam_lost_rotate_speed: float = 0.6
    stuck_time: float = 3.0
    min_progress: float = 0.2
    backup_time: float = 1.0
    backup_speed: float = -0.15
    rotate_time: float = 2.0
    rotate_speed: float = 0.6


@dataclass
class NavigatorRuntimeParams:
    control_hz: float = 10.0
    plan_hz: float = 1.0
    max_depth_age: float = 0.5
    replan_on_costmap: bool = True
    auto_goal_enable: bool = False
    auto_goal_distance: float = 2.0
    auto_goal_interval: float = 8.0


@dataclass
class TransformParams:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class TFParams:
    use_pose_topic: bool = False
    pose_topic: str = "/orbslam/pose"
    pose_child_frame: str = "base_link"
    pose_in_map_frame: bool = True
    publish_tf: bool = True
    publish_static_tf: bool = True
    pose_timeout: float = 0.5
    camera_in_base: TransformParams = field(default_factory=TransformParams)


@dataclass
class NavigatorParams:
    topics: Topics = field(default_factory=Topics)
    frames: Frames = field(default_factory=Frames)
    camera: CameraParams = field(default_factory=CameraParams)
    costmap: CostmapParams = field(default_factory=CostmapParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    recovery: RecoveryParams = field(default_factory=RecoveryParams)
    navigator: NavigatorRuntimeParams = field(default_factory=NavigatorRuntimeParams)
    tf: TFParams = field(default_factory=TFParams)


def _iter_params(prefix: str, obj: Any):
    for f in fields(obj):
        val = getattr(obj, f.name)
        name = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(val):
            yield from _iter_params(name, val)
        else:
            yield name, val


def _set_attr_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)


def declare_parameters(node, params: NavigatorParams) -> None:
    for name, val in _iter_params("", params):
        node.declare_parameter(name, val)


def params_from_node(node, params: NavigatorParams) -> NavigatorParams:
    out = copy.deepcopy(params)
    for name, default in _iter_params("", out):
        param = node.get_parameter(name)
        if param is not None and param.type_ != param.Type.NOT_SET:
            _set_attr_path(out, name, param.value)
        else:
            _set_attr_path(out, name, default)
    return out


def _update_dataclass(dc: Any, data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if not hasattr(dc, k):
            continue
        current = getattr(dc, k)
        if is_dataclass(current) and isinstance(v, dict):
            _update_dataclass(current, v)
        else:
            setattr(dc, k, v)


def load_params_from_yaml(path: str) -> NavigatorParams:
    params = NavigatorParams()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "navigator_node" in data:
        data = data.get("navigator_node", {})
    if "ros__parameters" in data:
        data = data.get("ros__parameters", {})

    if isinstance(data, dict):
        _update_dataclass(params, data)
    return params
