from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "D455SensorAdapter": ("adapters.sensors.d455_sensor", "D455SensorAdapter"),
    "D455SensorAdapterConfig": ("adapters.sensors.d455_sensor", "D455SensorAdapterConfig"),
    "NavDPClient": ("adapters.navdp_http", "NavDPClient"),
    "NavDPClientConfig": ("adapters.navdp_http", "NavDPClientConfig"),
    "NavDPClientError": ("adapters.navdp_http", "NavDPClientError"),
    "NavDPNoGoalResponse": ("adapters.navdp_http", "NavDPNoGoalResponse"),
    "NavDPPlannerState": ("adapters.navdp_http", "NavDPPlannerState"),
    "NavDPPointGoalResponse": ("adapters.navdp_http", "NavDPPointGoalResponse"),
    "is_valid_world_trajectory": ("adapters.navdp_http", "is_valid_world_trajectory"),
    "trajectory_robot_to_world": ("common.geometry", "trajectory_robot_to_world"),
    "world_goal_to_robot_frame": ("common.geometry", "world_goal_to_robot_frame"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
