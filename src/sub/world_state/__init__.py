from __future__ import annotations

from pathlib import Path
from typing import Final

from ..shared import load_module, root_path


NAME: Final[str] = "World State Subsystem"
DESCRIPTION: Final[str] = "Camera sensing, runtime state snapshots, and world/scene context used by navigation and planning."

MODULES: Final[dict[str, str]] = {
    "camera_api": "g1_play.camera_api",
    "camera_runtime": "g1_play.camera_runtime",
    "camera_control_api": "g1_play.camera_control.api",
    "camera_sensor": "g1_play.camera_control.sensor",
    "camera_service": "g1_play.camera_control.runtime_service",
    "runtime_state": "g1_play.navdp_runtime",
    "scene": "g1_play.scene",
    "paths": "g1_play.paths",
}

ENTRYPOINTS: Final[dict[str, Path]] = {
    "robot_asset": root_path("robots", "g1", "g1_d455.usd"),
    "runtime_config": root_path("tuned", "params", "env.yaml"),
}

PUBLIC_APIS: Final[tuple[str, ...]] = (
    "g1_play.camera_control.sensor.G1NavCameraSensor",
    "g1_play.camera_control.runtime_service.RuntimeCameraPitchService",
    "g1_play.navdp_runtime.NavigationPipelineState",
    "g1_play.navdp_runtime.TaskExecutionState",
    "g1_play.paths.repo_dir",
)


def load(alias: str):
    return load_module(MODULES[alias])
