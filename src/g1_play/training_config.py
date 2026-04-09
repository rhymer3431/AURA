"""Helpers for loading runtime defaults from exported training configs."""

from __future__ import annotations

from dataclasses import dataclass
import os

import yaml


@dataclass(slots=True)
class RuntimeTrainingConfig:
    """Runtime parameters extracted from an exported training config directory."""

    source_path: str
    physics_dt: float | None = None
    decimation: int | None = None
    robot_position: tuple[float, float, float] | None = None
    action_scale: float | None = None
    default_joint_pos_patterns: dict[str, float] | None = None
    stiffness_patterns: dict[str, float] | None = None
    damping_patterns: dict[str, float] | None = None
    solver_position_iterations: int | None = None
    solver_velocity_iterations: int | None = None
    height_scan_size: tuple[float, float] | None = None
    height_scan_resolution: float | None = None
    height_scan_enabled: bool | None = None


def _as_float_tuple(values) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values)


def _expand_pattern_values(joint_patterns, value_cfg) -> dict[str, float]:
    if value_cfg is None:
        return {}
    if isinstance(value_cfg, dict):
        return {str(pattern): float(value) for pattern, value in value_cfg.items()}
    if joint_patterns is None:
        return {}
    return {str(pattern): float(value_cfg) for pattern in joint_patterns}


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.unsafe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Training config must contain a top-level mapping: {path}")
    return data


def load_runtime_training_config(config_dir: str) -> RuntimeTrainingConfig:
    env_config_path = os.path.join(config_dir, "env.yaml")
    if not os.path.isfile(env_config_path):
        raise FileNotFoundError(f"Training env config not found: {env_config_path}")

    env_cfg = _load_yaml(env_config_path)
    scene_cfg = env_cfg.get("scene", {}) or {}
    robot_cfg = scene_cfg.get("robot", {}) or {}
    spawn_cfg = robot_cfg.get("spawn", {}) or {}
    init_state_cfg = robot_cfg.get("init_state", {}) or {}
    articulation_props = spawn_cfg.get("articulation_props", {}) or {}
    joint_action_cfg = (env_cfg.get("actions", {}) or {}).get("joint_pos", {}) or {}
    height_scanner_cfg = scene_cfg.get("height_scanner")

    stiffness_patterns: dict[str, float] = {}
    damping_patterns: dict[str, float] = {}
    for actuator_cfg in (robot_cfg.get("actuators", {}) or {}).values():
        if not isinstance(actuator_cfg, dict):
            continue
        joint_patterns = actuator_cfg.get("joint_names_expr") or actuator_cfg.get("joint_names")
        stiffness_patterns.update(_expand_pattern_values(joint_patterns, actuator_cfg.get("stiffness")))
        damping_patterns.update(_expand_pattern_values(joint_patterns, actuator_cfg.get("damping")))

    height_scan_size = None
    height_scan_resolution = None
    if isinstance(height_scanner_cfg, dict):
        pattern_cfg = height_scanner_cfg.get("pattern_cfg", {}) or {}
        size = _as_float_tuple(pattern_cfg.get("size"))
        if size is not None and len(size) == 2:
            height_scan_size = (float(size[0]), float(size[1]))
        resolution = pattern_cfg.get("resolution")
        if resolution is not None:
            height_scan_resolution = float(resolution)

    robot_position = _as_float_tuple(init_state_cfg.get("pos"))
    if robot_position is not None and len(robot_position) != 3:
        raise ValueError(f"Robot init position must have exactly three values: {env_config_path}")

    return RuntimeTrainingConfig(
        source_path=env_config_path,
        physics_dt=float(env_cfg["sim"]["dt"]) if env_cfg.get("sim", {}).get("dt") is not None else None,
        decimation=int(env_cfg["decimation"]) if env_cfg.get("decimation") is not None else None,
        robot_position=(float(robot_position[0]), float(robot_position[1]), float(robot_position[2]))
        if robot_position is not None
        else None,
        action_scale=float(joint_action_cfg["scale"]) if joint_action_cfg.get("scale") is not None else None,
        default_joint_pos_patterns={
            str(pattern): float(value) for pattern, value in (init_state_cfg.get("joint_pos", {}) or {}).items()
        }
        or None,
        stiffness_patterns=stiffness_patterns or None,
        damping_patterns=damping_patterns or None,
        solver_position_iterations=(
            int(articulation_props["solver_position_iteration_count"])
            if articulation_props.get("solver_position_iteration_count") is not None
            else None
        ),
        solver_velocity_iterations=(
            int(articulation_props["solver_velocity_iteration_count"])
            if articulation_props.get("solver_velocity_iteration_count") is not None
            else None
        ),
        height_scan_size=height_scan_size,
        height_scan_resolution=height_scan_resolution,
        height_scan_enabled=height_scanner_cfg is not None,
    )
