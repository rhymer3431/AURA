from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ipc.messages import FrameHeader, MessagePayload, message_to_dict


@dataclass(frozen=True)
class FrameCache:
    seq: int
    frame_header: FrameHeader
    rgb_image: np.ndarray
    depth_image_m: np.ndarray | None
    viewer_overlay: dict[str, object]
    last_frame_monotonic: float


@dataclass(frozen=True)
class GatewayEvent:
    kind: str
    payload: dict[str, Any]


def frame_age_ms(frame: FrameCache | None, *, now: float | None = None) -> float | None:
    if frame is None:
        return None
    current = time.monotonic() if now is None else float(now)
    return max((current - float(frame.last_frame_monotonic)) * 1000.0, 0.0)


def is_frame_stale(frame: FrameCache | None, *, stale_after_sec: float, now: float | None = None) -> bool:
    if frame is None:
        return True
    current = time.monotonic() if now is None else float(now)
    return (current - float(frame.last_frame_monotonic)) > max(float(stale_after_sec), 0.0)


def build_session_ready_message(
    *,
    session_id: str,
    track_roles: list[str],
    channel_labels: tuple[str, str],
) -> dict[str, object]:
    return {
        "type": "session_ready",
        "sessionId": str(session_id),
        "observeOnly": True,
        "trackRoles": list(track_roles),
        "channelLabels": list(channel_labels),
    }


def build_waiting_for_frame_message(*, age_ms: float | None, has_seen_frame: bool) -> dict[str, object]:
    return {
        "type": "waiting_for_frame",
        "age_ms": None if age_ms is None else round(float(age_ms), 3),
        "has_seen_frame": bool(has_seen_frame),
    }


def build_snapshot_message(frame: FrameCache, *, active_command_type: str = "") -> dict[str, object]:
    detections = frame.viewer_overlay.get("detections", [])
    snapshot_action = str(active_command_type).strip()
    active_target = frame.viewer_overlay.get("active_target", {})
    planner_overlay = frame.frame_header.metadata.get("planner_overlay", {})
    if isinstance(active_target, dict):
        maybe_action = str(active_target.get("action_type", "")).strip()
        if maybe_action != "":
            snapshot_action = maybe_action
    payload = {
        "type": "snapshot",
        "seq": int(frame.seq),
        "frame_id": int(frame.frame_header.frame_id),
        "source": str(frame.frame_header.source),
        "image": {
            "width": int(frame.frame_header.width),
            "height": int(frame.frame_header.height),
            "rgbEncoding": str(frame.frame_header.rgb_encoding),
        },
        "robot_pose_xyz": [float(value) for value in frame.frame_header.robot_pose_xyz[:3]],
        "robot_yaw_rad": float(frame.frame_header.robot_yaw_rad),
        "sim_time_s": float(frame.frame_header.sim_time_s),
        "detector_backend": str(frame.viewer_overlay.get("detector_backend", "")),
        "detection_count": len(detections) if isinstance(detections, list) else 0,
        "active_command_type": snapshot_action,
        "has_depth": frame.depth_image_m is not None,
    }
    if isinstance(active_target, dict) and active_target:
        payload["active_target"] = dict(active_target)
        payload["activeTarget"] = dict(active_target)
    if isinstance(planner_overlay, dict):
        for source_key, target_key in (
            ("plan_version", "planVersion"),
            ("goal_version", "goalVersion"),
            ("traj_version", "trajVersion"),
            ("stale_sec", "staleSec"),
            ("planner_control_mode", "plannerControlMode"),
            ("planner_yaw_delta_rad", "plannerYawDeltaRad"),
            ("interactive_phase", "interactivePhase"),
            ("interactive_command_id", "interactiveCommandId"),
            ("interactive_instruction", "interactiveInstruction"),
        ):
            value = planner_overlay.get(source_key)
            if value is None:
                continue
            payload[target_key] = value
    system2_pixel_goal = frame.viewer_overlay.get("system2_pixel_goal")
    if isinstance(system2_pixel_goal, list) and len(system2_pixel_goal) >= 2:
        compact_goal = [int(system2_pixel_goal[0]), int(system2_pixel_goal[1])]
        payload["system2_pixel_goal"] = compact_goal
        payload["system2PixelGoal"] = compact_goal
    return payload


def build_frame_meta_message(frame: FrameCache) -> dict[str, object]:
    overlay = frame.viewer_overlay if isinstance(frame.viewer_overlay, dict) else {}
    planner_overlay = frame.frame_header.metadata.get("planner_overlay", {})
    detections = overlay.get("detections", [])
    compact_detections: list[dict[str, object]] = []
    if isinstance(detections, list):
        for item in detections:
            if not isinstance(item, dict):
                continue
            compact: dict[str, object] = {}
            for key in ("class_name", "track_id"):
                value = item.get(key)
                if isinstance(value, str) and value != "":
                    compact[key] = value
            bbox = item.get("bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                compact["bbox_xyxy"] = [int(value) for value in bbox]
            for key in ("confidence", "depth_m", "approach_yaw_rad"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    compact[key] = float(value)
            world_pose = item.get("world_pose_xyz")
            if isinstance(world_pose, list) and len(world_pose) >= 3:
                compact["world_pose_xyz"] = [float(world_pose[0]), float(world_pose[1]), float(world_pose[2])]
            compact_detections.append(compact)

    trajectory_pixels = overlay.get("trajectory_pixels", [])
    compact_trajectory = []
    if isinstance(trajectory_pixels, list):
        for point in trajectory_pixels:
            if isinstance(point, list) and len(point) == 2:
                compact_trajectory.append([int(point[0]), int(point[1])])

    payload: dict[str, object] = {
        "type": "frame_meta",
        "seq": int(frame.seq),
        "frame_id": int(frame.frame_header.frame_id),
        "timestamp_ns": int(frame.frame_header.timestamp_ns),
        "source": str(frame.frame_header.source),
        "robot_pose_xyz": [float(value) for value in frame.frame_header.robot_pose_xyz[:3]],
        "robot_yaw_rad": float(frame.frame_header.robot_yaw_rad),
        "sim_time_s": float(frame.frame_header.sim_time_s),
        "detections": compact_detections,
        "trajectory_pixels": compact_trajectory,
        "trajectoryPixels": compact_trajectory,
    }
    active_target = overlay.get("active_target", {})
    if isinstance(active_target, dict):
        payload["active_target"] = dict(active_target)
        payload["activeTarget"] = dict(active_target)
    if isinstance(planner_overlay, dict):
        for source_key, target_key in (
            ("plan_version", "planVersion"),
            ("goal_version", "goalVersion"),
            ("traj_version", "trajVersion"),
            ("stale_sec", "staleSec"),
            ("planner_control_mode", "plannerControlMode"),
            ("planner_yaw_delta_rad", "plannerYawDeltaRad"),
            ("interactive_phase", "interactivePhase"),
            ("interactive_command_id", "interactiveCommandId"),
            ("interactive_instruction", "interactiveInstruction"),
        ):
            value = planner_overlay.get(source_key)
            if value is None:
                continue
            payload[target_key] = value
    system2_pixel_goal = overlay.get("system2_pixel_goal")
    if isinstance(system2_pixel_goal, list) and len(system2_pixel_goal) >= 2:
        compact_goal = [int(system2_pixel_goal[0]), int(system2_pixel_goal[1])]
        payload["system2_pixel_goal"] = compact_goal
        payload["system2PixelGoal"] = compact_goal
    return payload


def ipc_message_event(kind: str, message: MessagePayload) -> dict[str, object]:
    payload = dict(message_to_dict(message))
    payload.pop("__type__", None)
    payload["type"] = str(kind)
    return payload
