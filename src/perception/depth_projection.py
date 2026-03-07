from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.geometry import project_camera_point_to_world
from inference.detectors.base import DetectionResult
from inference.trackers.simple_tracker import TrackedDetection


@dataclass(frozen=True)
class ProjectedDetection:
    class_name: str
    confidence: float
    world_pose_xyz: tuple[float, float, float]
    track_id: str
    room_id: str = ""
    movable: bool = True
    state: str = "visible"
    embedding_id: str = ""
    snapshots: list[str] | None = None
    metadata: dict[str, Any] | None = None


class DepthProjector:
    def project(
        self,
        tracked_detection: TrackedDetection,
        *,
        depth_image_m: np.ndarray,
        camera_intrinsic: np.ndarray,
        camera_pose_xyz: tuple[float, float, float] | np.ndarray,
        camera_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectedDetection | None:
        metadata = dict(metadata or {})
        detection = tracked_detection.detection
        centroid = detection.centroid_xy or self._bbox_center(detection.bbox_xyxy)
        depth_value = self._resolve_depth(detection, centroid, np.asarray(depth_image_m, dtype=np.float32))
        if depth_value <= 0.0:
            depth_value = float(metadata.get("default_depth_m", 1.0))
        camera_point = self._project_pixel_to_camera(
            centroid_xy=centroid,
            depth_m=depth_value,
            intrinsic=np.asarray(camera_intrinsic, dtype=np.float32),
        )
        world_point = project_camera_point_to_world(
            point_xyz_camera=camera_point,
            camera_pos_world=np.asarray(camera_pose_xyz, dtype=np.float32),
            camera_quat_wxyz=np.asarray(camera_quat_wxyz, dtype=np.float32),
        )
        return ProjectedDetection(
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            world_pose_xyz=(float(world_point[0]), float(world_point[1]), float(world_point[2])),
            track_id=tracked_detection.track_id,
            room_id=str(metadata.get("room_id", "")),
            movable=bool(metadata.get("movable", detection.class_name.lower() not in {"table", "chair", "wall"})),
            state=str(metadata.get("state", "visible")),
            embedding_id=str(detection.metadata.get("embedding_id", metadata.get("embedding_id", ""))),
            snapshots=list(metadata.get("snapshots", [])),
            metadata={
                **detection.metadata,
                **metadata,
                "depth_m": float(depth_value),
                "bbox_xyxy": list(detection.bbox_xyxy),
                "centroid_xy": [float(centroid[0]), float(centroid[1])],
                "appearance_signature": detection.metadata.get("appearance_signature", metadata.get("appearance_signature", "")),
            },
        )

    @staticmethod
    def _bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
        x0, y0, x1, y1 = bbox_xyxy
        return ((float(x0) + float(x1)) * 0.5, (float(y0) + float(y1)) * 0.5)

    @staticmethod
    def _resolve_depth(detection: DetectionResult, centroid_xy: tuple[float, float], depth_image: np.ndarray) -> float:
        if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
            depth_image = depth_image[..., 0]
        if depth_image.ndim != 2:
            raise ValueError(f"depth_image_m must be [H,W] or [H,W,1], got {depth_image.shape}")
        x = int(np.clip(round(float(centroid_xy[0])), 0, depth_image.shape[1] - 1))
        y = int(np.clip(round(float(centroid_xy[1])), 0, depth_image.shape[0] - 1))
        if detection.mask is not None and detection.mask.shape[:2] == depth_image.shape[:2]:
            masked = depth_image[detection.mask.astype(bool)]
            valid = masked[np.isfinite(masked) & (masked > 0.0)]
            if valid.size > 0:
                return float(np.median(valid))
        x0, y0, x1, y1 = detection.bbox_xyxy
        roi = depth_image[max(y0, 0) : min(y1 + 1, depth_image.shape[0]), max(x0, 0) : min(x1 + 1, depth_image.shape[1])]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size > 0:
            return float(np.median(valid))
        value = depth_image[y, x]
        if not np.isfinite(value):
            return 0.0
        return float(value)

    @staticmethod
    def _project_pixel_to_camera(centroid_xy: tuple[float, float], depth_m: float, intrinsic: np.ndarray) -> np.ndarray:
        if intrinsic.shape[0] < 3 or intrinsic.shape[1] < 3:
            raise ValueError(f"intrinsic must be 3x3, got {intrinsic.shape}")
        u = float(centroid_xy[0])
        v = float(centroid_xy[1])
        fx = float(intrinsic[0, 0]) if float(intrinsic[0, 0]) != 0.0 else 1.0
        fy = float(intrinsic[1, 1]) if float(intrinsic[1, 1]) != 0.0 else 1.0
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        x = (u - cx) * float(depth_m) / fx
        y = (v - cy) * float(depth_m) / fy
        return np.asarray([x, y, float(depth_m)], dtype=np.float32)
