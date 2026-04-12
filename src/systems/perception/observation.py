"""Observation normalization owned by the perception subsystem."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from systems.shared.contracts.observation import ObservationFrame, RawObservation

from .telemetry import ViewerFramePublisher


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {image.shape}.")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {image.shape}.")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(image)


class PerceptionObservationService:
    """Normalize raw Isaac captures and publish viewer-compatible frames."""

    def __init__(self, *, viewer_publisher: ViewerFramePublisher | None = None):
        self._viewer_publisher = viewer_publisher
        self._latest_health: dict[str, Any] = {
            "status": "idle",
            "frame_id": None,
            "last_error": None,
            "last_stamp_s": None,
        }

    def ingest(self, raw: RawObservation) -> ObservationFrame:
        frame = ObservationFrame(
            rgb=_normalize_rgb(raw.rgb),
            depth=_normalize_depth(raw.depth),
            intrinsic=np.asarray(raw.intrinsic, dtype=np.float32).copy(),
            camera_pos_w=np.asarray(raw.camera_pos_w, dtype=np.float32).copy(),
            camera_rot_w=np.asarray(raw.camera_rot_w, dtype=np.float32).copy(),
            robot_state=raw.robot_state,
            stamp_s=float(raw.stamp_s),
            metadata=dict(raw.metadata),
        )
        if self._viewer_publisher is not None:
            robot_pose_xyz = np.asarray(raw.robot_state.base_pos_w, dtype=np.float32)
            self._latest_health = {
                "status": "running",
                "last_error": None,
                "last_stamp_s": frame.stamp_s,
                **self._viewer_publisher.publish_frame(
                    rgb=frame.rgb,
                    depth=frame.depth,
                    source="perception_runtime",
                    frame_stamp_s=frame.stamp_s,
                    camera_pos_w=frame.camera_pos_w,
                    camera_rot_w=frame.camera_rot_w,
                    robot_pose_xyz=robot_pose_xyz,
                    robot_yaw_rad=float(raw.robot_state.base_yaw),
                    intrinsic=frame.intrinsic,
                    metadata=frame.metadata,
                ),
            }
        else:
            self._latest_health = {
                "status": "running",
                "frame_id": None,
                "last_error": None,
                "last_stamp_s": frame.stamp_s,
            }
        return frame

    def latest_health(self) -> dict[str, Any]:
        payload = dict(self._latest_health)
        last_stamp_s = payload.get("last_stamp_s")
        if isinstance(last_stamp_s, (int, float)):
            payload["frame_age_ms"] = max(0.0, (time.monotonic() - float(last_stamp_s)) * 1000.0)
        return payload

    def close(self) -> None:
        if self._viewer_publisher is not None:
            self._viewer_publisher.close()
            self._viewer_publisher = None

