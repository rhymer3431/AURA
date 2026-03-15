from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locomotion.constants import ACTION_SCALE, DEFAULT_DECIMATION, DEFAULT_PHYSICS_DT


PROCESS_NAMES = ("navdp", "system2", "dual", "runtime")


@dataclass(frozen=True)
class DashboardLocomotionConfig:
    action_scale: float = ACTION_SCALE
    onnx_device: str = "auto"
    physics_dt: float = DEFAULT_PHYSICS_DT
    decimation: int = DEFAULT_DECIMATION
    rendering_dt: float = 0.0
    cmd_vel_timeout: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "actionScale": float(self.action_scale),
            "onnxDevice": self.onnx_device,
            "physicsDt": float(self.physics_dt),
            "decimation": int(self.decimation),
            "renderingDt": float(self.rendering_dt),
            "cmdVelTimeout": float(self.cmd_vel_timeout),
        }


@dataclass(frozen=True)
class DashboardSessionRequest:
    planner_mode: str
    launch_mode: str
    scene_preset: str
    viewer_enabled: bool
    memory_store: bool
    detection_enabled: bool
    locomotion_config: DashboardLocomotionConfig = DashboardLocomotionConfig()
    goal_x: float | None = None
    goal_y: float | None = None

    def required_process_names(self) -> set[str]:
        required = {"navdp", "runtime"}
        if self.planner_mode == "interactive":
            required.update({"system2", "dual"})
        return required

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "plannerMode": self.planner_mode,
            "launchMode": self.launch_mode,
            "scenePreset": self.scene_preset,
            "viewerEnabled": self.viewer_enabled,
            "memoryStore": self.memory_store,
            "detectionEnabled": self.detection_enabled,
            "locomotionConfig": self.locomotion_config.to_public_dict(),
        }
        if self.goal_x is not None and self.goal_y is not None:
            payload["goal"] = {"x": float(self.goal_x), "y": float(self.goal_y)}
        return payload


def _parse_locomotion_config(payload: Any) -> DashboardLocomotionConfig:
    config_payload = payload if isinstance(payload, dict) else {}
    onnx_device = str(config_payload.get("onnxDevice", "auto")).strip().lower() or "auto"
    if onnx_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("locomotionConfig.onnxDevice must be auto, cuda, or cpu")
    try:
        action_scale = float(config_payload.get("actionScale", ACTION_SCALE))
        physics_dt = float(config_payload.get("physicsDt", DEFAULT_PHYSICS_DT))
        decimation = int(config_payload.get("decimation", DEFAULT_DECIMATION))
        rendering_dt = float(config_payload.get("renderingDt", 0.0))
        cmd_vel_timeout = float(config_payload.get("cmdVelTimeout", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "locomotionConfig.actionScale/physicsDt/renderingDt/cmdVelTimeout must be numbers and decimation must be an integer"
        ) from exc
    if action_scale <= 0.0:
        raise ValueError("locomotionConfig.actionScale must be positive")
    if physics_dt <= 0.0:
        raise ValueError("locomotionConfig.physicsDt must be positive")
    if decimation <= 0:
        raise ValueError("locomotionConfig.decimation must be positive")
    if rendering_dt < 0.0:
        raise ValueError("locomotionConfig.renderingDt must be non-negative")
    return DashboardLocomotionConfig(
        action_scale=action_scale,
        onnx_device=onnx_device,
        physics_dt=physics_dt,
        decimation=decimation,
        rendering_dt=rendering_dt,
        cmd_vel_timeout=cmd_vel_timeout,
    )


def parse_session_request(payload: dict[str, Any]) -> DashboardSessionRequest:
    planner_mode = str(payload.get("plannerMode", "")).strip().lower()
    launch_mode = str(payload.get("launchMode", "")).strip().lower()
    scene_preset = str(payload.get("scenePreset", "warehouse")).strip() or "warehouse"
    viewer_enabled = bool(payload.get("viewerEnabled", True))
    memory_store = bool(payload.get("memoryStore", True))
    detection_enabled = bool(payload.get("detectionEnabled", True))
    locomotion_config = _parse_locomotion_config(payload.get("locomotionConfig"))
    if planner_mode not in {"interactive", "pointgoal"}:
        raise ValueError("plannerMode must be interactive or pointgoal")
    if launch_mode not in {"gui", "headless"}:
        raise ValueError("launchMode must be gui or headless")
    goal_payload = payload.get("goal")
    goal_x: float | None = None
    goal_y: float | None = None
    if planner_mode == "pointgoal":
        if not isinstance(goal_payload, dict):
            raise ValueError("goal is required when plannerMode=pointgoal")
        try:
            goal_x = float(goal_payload.get("x"))
            goal_y = float(goal_payload.get("y"))
        except (TypeError, ValueError) as exc:
            raise ValueError("goal.x and goal.y must be numbers") from exc
    return DashboardSessionRequest(
        planner_mode=planner_mode,
        launch_mode=launch_mode,
        scene_preset=scene_preset,
        viewer_enabled=viewer_enabled,
        memory_store=memory_store,
        detection_enabled=detection_enabled,
        locomotion_config=locomotion_config,
        goal_x=goal_x,
        goal_y=goal_y,
    )


def to_iso_timestamp(epoch_sec: float | None) -> str | None:
    if epoch_sec is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(epoch_sec), tz=timezone.utc).isoformat()


def resolve_repo_path(repo_root: Path, *parts: str) -> Path:
    return repo_root.joinpath(*parts).resolve()
