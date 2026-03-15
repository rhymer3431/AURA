from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROCESS_NAMES = ("navdp", "system2", "dual", "runtime")


@dataclass(frozen=True)
class DashboardSessionRequest:
    planner_mode: str
    launch_mode: str
    scene_preset: str
    viewer_enabled: bool
    memory_store: bool
    detection_enabled: bool
    policy_path: str | None = None
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
        }
        if self.policy_path is not None:
            payload["policyPath"] = self.policy_path
        if self.goal_x is not None and self.goal_y is not None:
            payload["goal"] = {"x": float(self.goal_x), "y": float(self.goal_y)}
        return payload


def parse_session_request(payload: dict[str, Any]) -> DashboardSessionRequest:
    planner_mode = str(payload.get("plannerMode", "")).strip().lower()
    launch_mode = str(payload.get("launchMode", "")).strip().lower()
    scene_preset = str(payload.get("scenePreset", "warehouse")).strip() or "warehouse"
    viewer_enabled = bool(payload.get("viewerEnabled", True))
    memory_store = bool(payload.get("memoryStore", True))
    detection_enabled = bool(payload.get("detectionEnabled", True))
    raw_policy_path = payload.get("policyPath")
    policy_path = str(raw_policy_path).strip() if raw_policy_path is not None else ""
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
        policy_path=policy_path or None,
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
