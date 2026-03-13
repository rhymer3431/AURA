from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from common.cv2_compat import cv2
from common.geometry import trajectory_robot_to_world, world_goal_to_robot_frame


class NavDPClientError(RuntimeError):
    """Raised when a NavDP HTTP request fails."""


@dataclass
class NavDPClientConfig:
    base_url: str = "http://127.0.0.1:8888"
    timeout_sec: float = 5.0
    reset_timeout_sec: float | None = 15.0
    retry: int = 1
    stop_threshold: float = -3.0


@dataclass
class NavDPPointGoalResponse:
    trajectory: np.ndarray
    all_trajectory: np.ndarray
    all_values: np.ndarray
    server_input_meta: dict[str, Any] | None = None


@dataclass
class NavDPNoGoalResponse:
    trajectory: np.ndarray
    all_trajectory: np.ndarray
    all_values: np.ndarray


@dataclass
class NavDPPlannerState:
    initialized: bool = False
    failed_calls: int = 0
    successful_calls: int = 0
    last_plan_step: int = -1
    last_waypoint_idx: int = 0
    last_trajectory_world: np.ndarray | None = None


class NavDPClient:
    def __init__(self, config: NavDPClientConfig):
        self.config = config

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request_timeout(self, path: str) -> float:
        route = path.strip("/").lower()
        if route == "navigator_reset":
            reset_timeout = self.config.reset_timeout_sec
            if reset_timeout is not None and float(reset_timeout) > 0.0:
                return float(reset_timeout)
        return float(self.config.timeout_sec)

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        payload = {
            "intrinsic": np.asarray(intrinsic, dtype=np.float32).tolist(),
            "stop_threshold": float(self.config.stop_threshold),
            "batch_size": int(batch_size),
        }
        last_error: Exception | None = None
        for _ in range(int(self.config.retry) + 1):
            try:
                resp = requests.post(
                    self._url("navigator_reset"),
                    json=payload,
                    timeout=self._request_timeout("navigator_reset"),
                )
                resp.raise_for_status()
                data = resp.json()
                algo = str(data.get("algo", ""))
                if algo == "":
                    raise NavDPClientError("navigator_reset returned empty algo.")
                return algo
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise NavDPClientError(f"navigator_reset failed after retries: {last_error}")

    @staticmethod
    def _normalize_depth_batch(depth_images_m: np.ndarray) -> np.ndarray:
        depths = np.asarray(depth_images_m, dtype=np.float32)
        if depths.ndim not in (3, 4):
            raise ValueError(f"depth_images_m must be [B,H,W] or [B,H,W,1], got shape={depths.shape}")
        if depths.ndim == 4:
            depths = depths[..., 0]
        return depths

    @staticmethod
    def _encode_rgbd_batch(rgb_images: np.ndarray, depth_images_m: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
        rgbs = np.asarray(rgb_images)
        depths = NavDPClient._normalize_depth_batch(depth_images_m)

        if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
            raise ValueError(f"rgb_images must be [B,H,W,3], got shape={rgbs.shape}")
        if rgbs.shape[0] != depths.shape[0]:
            raise ValueError(f"batch mismatch: rgbs={rgbs.shape[0]} depths={depths.shape[0]}")

        concat_images = np.concatenate([img for img in rgbs], axis=0)
        concat_depths = np.concatenate([img for img in depths], axis=0)
        concat_depths = np.nan_to_num(concat_depths, nan=0.0, posinf=0.0, neginf=0.0)

        ok_rgb, rgb_buf = cv2.imencode(".jpg", concat_images)
        if not ok_rgb:
            raise NavDPClientError("Failed to encode RGB image as JPEG.")
        depth_mm_u16 = np.clip(concat_depths * 10000.0, 0.0, 65535.0).astype(np.uint16)
        ok_depth, depth_buf = cv2.imencode(".png", depth_mm_u16)
        if not ok_depth:
            raise NavDPClientError("Failed to encode depth image as PNG.")

        files = {
            "image": ("image.jpg", io.BytesIO(rgb_buf).getvalue(), "image/jpeg"),
            "depth": ("depth.png", io.BytesIO(depth_buf).getvalue(), "image/png"),
        }
        client_meta = {
            "batch_size": int(rgbs.shape[0]),
            "rgb_concat_shape": list(concat_images.shape),
            "depth_concat_shape": list(concat_depths.shape),
            "depth_min_m": float(np.min(concat_depths)) if concat_depths.size > 0 else 0.0,
            "depth_max_m": float(np.max(concat_depths)) if concat_depths.size > 0 else 0.0,
        }
        return files, client_meta, concat_depths

    def _request_trajectory_step(
        self,
        *,
        route: str,
        files: dict[str, Any],
        data: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        last_error: Exception | None = None
        for _ in range(int(self.config.retry) + 1):
            try:
                resp = requests.post(
                    self._url(route),
                    files=files,
                    data=data,
                    timeout=self._request_timeout(route),
                )
                resp.raise_for_status()
                body: dict[str, Any] = resp.json()
                trajectory = np.asarray(body.get("trajectory"), dtype=np.float32)
                all_trajectory = np.asarray(body.get("all_trajectory"), dtype=np.float32)
                all_values = np.asarray(body.get("all_values"), dtype=np.float32)
                if trajectory.size == 0:
                    raise NavDPClientError(f"{route} returned an empty trajectory.")
                return trajectory, all_trajectory, all_values, body
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise NavDPClientError(f"{route} failed after retries: {last_error}")

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        goals = np.asarray(point_goals, dtype=np.float32)
        rgbs = np.asarray(rgb_images)
        depths = self._normalize_depth_batch(depth_images_m)

        if goals.ndim != 2 or goals.shape[1] < 2:
            raise ValueError(f"point_goals must be [B,2+], got shape={goals.shape}")
        if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
            raise ValueError(f"rgb_images must be [B,H,W,3], got shape={rgbs.shape}")
        if rgbs.shape[0] != goals.shape[0] or depths.shape[0] != goals.shape[0]:
            raise ValueError(
                "batch mismatch: "
                f"goals={goals.shape[0]} rgbs={rgbs.shape[0]} depths={depths.shape[0]}"
            )

        files, client_meta, _ = self._encode_rgbd_batch(rgbs, depths)
        goal_payload: dict[str, Any] = {
            "goal_x": goals[:, 0].astype(np.float32).tolist(),
            "goal_y": goals[:, 1].astype(np.float32).tolist(),
            "client_meta": client_meta,
        }
        if sensor_meta is not None:
            goal_payload["sensor_meta"] = sensor_meta
        data = {"goal_data": json.dumps(goal_payload)}

        trajectory, all_trajectory, all_values, body = self._request_trajectory_step(
            route="pointgoal_step",
            files=files,
            data=data,
        )
        return NavDPPointGoalResponse(
            trajectory=trajectory,
            all_trajectory=all_trajectory,
            all_values=all_values,
            server_input_meta=body.get("input_meta"),
        )

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        files, _, _ = self._encode_rgbd_batch(rgb_images, depth_images_m)
        trajectory, all_trajectory, all_values, _ = self._request_trajectory_step(
            route="nogoal_step",
            files=files,
        )
        return NavDPNoGoalResponse(
            trajectory=trajectory,
            all_trajectory=all_trajectory,
            all_values=all_values,
        )


def is_valid_world_trajectory(path_xyz: np.ndarray) -> bool:
    arr = np.asarray(path_xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return False
    return bool(np.all(np.isfinite(arr)))
