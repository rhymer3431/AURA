from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class SlamMode:
    LOCALIZATION = "localization"
    EXPLORATION = "exploration"


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"
    covariance_norm: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class Detection2D3D:
    object_id: str
    class_name: str
    score: float
    bbox_xywh: Optional[Tuple[float, float, float, float]] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    bbox_cxcy: Optional[Tuple[float, float]] = None
    image_size: Optional[Tuple[int, int]] = None
    frame_id: str = ""
    mask: Optional[Any] = None
    position_in_map: Optional[Pose2D] = None
    timestamp: float = field(default_factory=lambda: time.time())

    @property
    def label(self) -> str:
        return self.class_name


@dataclass
class ObjectMemoryEntry:
    object_id: str
    class_name: str
    map_pose: Pose2D
    last_seen: float
    confidence: float
    importance: float


@dataclass
class RetryPolicy:
    max_retries: int = 1
    backoff_s: float = 1.0


@dataclass
class SkillCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass
class Plan:
    skills: List[SkillCall] = field(default_factory=list)
    notes: str = ""


@dataclass
class NavResult:
    success: bool
    reason: str = "OK"


def pose_from_dict(payload: Dict[str, Any]) -> Pose2D:
    return Pose2D(
        x=float(payload.get("x", 0.0)),
        y=float(payload.get("y", 0.0)),
        yaw=float(payload.get("yaw", 0.0)),
        frame_id=str(payload.get("frame_id", "map")),
    )


def pose_to_dict(pose: Optional[Pose2D]) -> Optional[Dict[str, Any]]:
    if pose is None:
        return None
    return {
        "x": pose.x,
        "y": pose.y,
        "yaw": pose.yaw,
        "frame_id": pose.frame_id,
        "covariance_norm": pose.covariance_norm,
        "timestamp": pose.timestamp,
    }


def plan_from_json(payload: Dict[str, Any]) -> Plan:
    skills: List[SkillCall] = []
    for raw_skill in payload.get("plan", []):
        retry = raw_skill.get("retry_policy") or {}
        skills.append(
            SkillCall(
                name=str(raw_skill.get("skill", "")).strip().lower(),
                args=dict(raw_skill.get("args") or {}),
                success_criteria=str(raw_skill.get("success_criteria", "")),
                retry_policy=RetryPolicy(
                    max_retries=max(0, int(retry.get("max_retries", 1))),
                    backoff_s=max(0.0, float(retry.get("backoff_s", 1.0))),
                ),
            )
        )
    return Plan(skills=skills, notes=str(payload.get("notes", "")))
