from __future__ import annotations

from typing import Any

import numpy as np

from common.geometry import quat_wxyz_to_rot_matrix

from .pipeline import PerceptionFrameResult


def _rounded_xyz(values: tuple[float, float, float] | np.ndarray | list[float]) -> list[float] | None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [round(float(arr[0]), 4), round(float(arr[1]), 4), round(float(arr[2]), 4)]


def _project_world_to_pixel(
    point_world: np.ndarray,
    *,
    camera_pose_xyz: tuple[float, float, float] | np.ndarray,
    camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
    camera_intrinsic: np.ndarray,
) -> list[int] | None:
    point = np.asarray(point_world, dtype=np.float32).reshape(-1)
    if point.shape[0] < 3:
        return None
    cam_pos = np.asarray(camera_pose_xyz, dtype=np.float32).reshape(-1)
    intrinsic = np.asarray(camera_intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        return None

    rotation = quat_wxyz_to_rot_matrix(np.asarray(camera_quat_wxyz, dtype=np.float32)).astype(np.float32)
    point_cam = rotation.T @ (point[:3] - cam_pos[:3])
    depth = float(point_cam[2])
    if depth <= 1.0e-4 or not np.isfinite(depth):
        return None

    fx = float(intrinsic[0, 0]) if float(intrinsic[0, 0]) != 0.0 else 1.0
    fy = float(intrinsic[1, 1]) if float(intrinsic[1, 1]) != 0.0 else 1.0
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    pixel_x = fx * float(point_cam[0]) / depth + cx
    pixel_y = fy * float(point_cam[1]) / depth + cy
    if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
        return None
    return [int(round(pixel_x)), int(round(pixel_y))]


def build_viewer_overlay_payload(
    frame_result: PerceptionFrameResult,
    *,
    max_detections: int = 64,
    camera_intrinsic: np.ndarray | None = None,
    camera_pose_xyz: tuple[float, float, float] | np.ndarray | None = None,
    camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray | None = None,
) -> dict[str, object]:
    projected_by_track: dict[str, dict[str, object]] = {}
    for projected in frame_result.projected_detections:
        track_id = str(projected.track_id)
        entry: dict[str, object] = {}
        metadata = projected.metadata or {}
        depth_value = metadata.get("depth_m")
        if isinstance(depth_value, (int, float)):
            entry["depth_m"] = round(float(depth_value), 4)
        world_pose_xyz = _rounded_xyz(projected.world_pose_xyz)
        if world_pose_xyz is not None:
            entry["world_pose_xyz"] = world_pose_xyz
        approach_yaw = metadata.get("approach_yaw_rad")
        if isinstance(approach_yaw, (int, float)):
            entry["approach_yaw_rad"] = round(float(approach_yaw), 6)
        projected_by_track[track_id] = entry

    overlay_detections: list[dict[str, object]] = []
    for tracked in frame_result.tracked_detections[: max(int(max_detections), 0)]:
        detection = tracked.detection
        overlay_entry: dict[str, object] = {
            "class_name": str(detection.class_name),
            "confidence": round(float(detection.confidence), 6),
            "bbox_xyxy": [int(value) for value in detection.bbox_xyxy],
            "track_id": str(tracked.track_id),
        }
        overlay_entry.update(projected_by_track.get(str(tracked.track_id), {}))
        overlay_detections.append(overlay_entry)

    metadata: dict[str, Any] = dict(frame_result.metadata)
    payload: dict[str, object] = {
        "detector_backend": str(metadata.get("detector", "")),
        "detector_selected_reason": str(metadata.get("detector_selected_reason", "")),
        "detections": overlay_detections,
    }
    planner_overlay = metadata.get("planner_overlay")
    if (
        isinstance(planner_overlay, dict)
        and camera_intrinsic is not None
        and camera_pose_xyz is not None
        and camera_quat_wxyz is not None
    ):
        raw_trajectory = np.asarray(planner_overlay.get("trajectory_world", []), dtype=np.float32)
        if raw_trajectory.ndim == 1 and raw_trajectory.size == 0:
            raw_trajectory = np.zeros((0, 3), dtype=np.float32)
        if raw_trajectory.ndim == 2 and raw_trajectory.shape[1] >= 3:
            trajectory_pixels: list[list[int]] = []
            for point_world in raw_trajectory:
                pixel = _project_world_to_pixel(
                    point_world,
                    camera_pose_xyz=camera_pose_xyz,
                    camera_quat_wxyz=camera_quat_wxyz,
                    camera_intrinsic=np.asarray(camera_intrinsic, dtype=np.float32),
                )
                if pixel is not None:
                    trajectory_pixels.append(pixel)
            if trajectory_pixels:
                payload["trajectory_pixels"] = trajectory_pixels
                payload["trajectory_point_count"] = len(trajectory_pixels)
                for key in ("plan_version", "goal_version", "traj_version"):
                    value = planner_overlay.get(key)
                    if isinstance(value, (int, float)) and int(value) >= 0:
                        payload[key] = int(value)
        pixel_goal = np.asarray(planner_overlay.get("system2_pixel_goal", []), dtype=np.float32).reshape(-1)
        if pixel_goal.shape[0] >= 2 and np.all(np.isfinite(pixel_goal[:2])):
            payload["system2_pixel_goal"] = [
                int(round(float(pixel_goal[0]))),
                int(round(float(pixel_goal[1]))),
            ]
    active_command_overlay = metadata.get("active_command_overlay")
    if isinstance(active_command_overlay, dict):
        active_target: dict[str, object] = {}
        for key in ("action_type", "target_mode", "target_class", "target_track_id", "pose_source"):
            value = active_command_overlay.get(key)
            if isinstance(value, str) and value != "":
                active_target[key] = value
        for key in ("approach_yaw_rad", "track_age_sec", "depth_m"):
            value = active_command_overlay.get(key)
            if isinstance(value, (int, float)):
                active_target[key] = round(float(value), 6)
        for key in ("raw_target_pose_xyz", "filtered_target_pose_xyz", "nav_goal_pose_xyz"):
            value = active_command_overlay.get(key)
            rounded_xyz = _rounded_xyz(value) if value is not None else None
            if rounded_xyz is not None:
                active_target[key] = rounded_xyz
        if (
            camera_intrinsic is not None
            and camera_pose_xyz is not None
            and camera_quat_wxyz is not None
        ):
            for pose_key, pixel_key in (
                ("raw_target_pose_xyz", "raw_target_pixel"),
                ("filtered_target_pose_xyz", "filtered_target_pixel"),
                ("nav_goal_pose_xyz", "nav_goal_pixel"),
            ):
                value = active_command_overlay.get(pose_key)
                if value is None:
                    continue
                pixel = _project_world_to_pixel(
                    np.asarray(value, dtype=np.float32),
                    camera_pose_xyz=camera_pose_xyz,
                    camera_quat_wxyz=camera_quat_wxyz,
                    camera_intrinsic=np.asarray(camera_intrinsic, dtype=np.float32),
                )
                if pixel is not None:
                    active_target[pixel_key] = pixel
        if active_target:
            payload["active_target"] = active_target
    return payload


__all__ = ["build_viewer_overlay_payload"]
