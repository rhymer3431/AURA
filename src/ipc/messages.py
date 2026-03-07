from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def _message_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    timestamp_ns: int
    source: str
    rgb_shm: str = ""
    depth_shm: str = ""
    width: int = 0
    height: int = 0
    rgb_encoding: str = "rgb8"
    depth_encoding: str = "32FC1"
    camera_pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: _message_id("frame"))


@dataclass(frozen=True)
class ActionCommand:
    action_type: Literal[
        "STOP",
        "LOOK_AT",
        "FOLLOW_PERSON",
        "NAV_TO_PLACE",
        "NAV_TO_POSE",
        "LOCAL_SEARCH",
    ]
    command_id: str = field(default_factory=lambda: _message_id("cmd"))
    task_id: str = ""
    target_object_id: str = ""
    target_place_id: str = ""
    target_track_id: str = ""
    target_pose_xyz: tuple[float, float, float] | None = None
    look_at_yaw_rad: float | None = None
    stop_radius_m: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionStatus:
    command_id: str
    state: Literal["idle", "running", "succeeded", "failed", "stale"]
    timestamp_ns: int = field(default_factory=time.time_ns)
    success: bool = False
    reason: str = ""
    robot_pose_xyz: tuple[float, float, float] | None = None
    distance_remaining_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRequest:
    command_text: str
    task_id: str = field(default_factory=lambda: _message_id("task"))
    intent: str = ""
    target_json: dict[str, Any] = field(default_factory=dict)
    speaker_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


MessagePayload = FrameHeader | ActionCommand | ActionStatus | TaskRequest


MESSAGE_TYPES: dict[str, type[MessagePayload]] = {
    "FrameHeader": FrameHeader,
    "ActionCommand": ActionCommand,
    "ActionStatus": ActionStatus,
    "TaskRequest": TaskRequest,
}


def message_to_dict(message: MessagePayload) -> dict[str, Any]:
    payload = asdict(message)
    payload["__type__"] = type(message).__name__
    return payload


def message_from_dict(payload: dict[str, Any]) -> MessagePayload:
    raw_type = str(payload.get("__type__", "")).strip()
    if raw_type == "":
        raise ValueError("IPC payload is missing __type__.")
    cls = MESSAGE_TYPES.get(raw_type)
    if cls is None:
        raise ValueError(f"Unsupported IPC message type: {raw_type}")
    kwargs = dict(payload)
    kwargs.pop("__type__", None)
    if cls is FrameHeader:
        kwargs["camera_pose_xyz"] = tuple(kwargs.get("camera_pose_xyz", (0.0, 0.0, 0.0)))
        kwargs["camera_quat_wxyz"] = tuple(kwargs.get("camera_quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
    elif cls is ActionCommand and kwargs.get("target_pose_xyz") is not None:
        kwargs["target_pose_xyz"] = tuple(kwargs["target_pose_xyz"])
    elif cls is ActionStatus and kwargs.get("robot_pose_xyz") is not None:
        kwargs["robot_pose_xyz"] = tuple(kwargs["robot_pose_xyz"])
    return cls(**kwargs)
