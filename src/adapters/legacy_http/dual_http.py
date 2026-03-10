from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from common.cv2_compat import cv2

class DualSystemClientError(RuntimeError):
    """Raised when dual-server HTTP request fails."""


@dataclass
class DualSystemClientConfig:
    base_url: str = "http://127.0.0.1:8890"
    timeout_sec: float = 5.0
    reset_timeout_sec: float | None = 15.0
    retry: int = 1


@dataclass
class DualResetResponse:
    algo: str
    state: dict[str, Any]


@dataclass
class DualStepResponse:
    trajectory_world: np.ndarray
    pixel_goal: np.ndarray | None
    stop: bool
    goal_version: int
    traj_version: int
    used_cached_traj: bool
    stale_sec: float
    planner_control: dict[str, Any]
    debug: dict[str, Any]


class DualSystemClient:
    def __init__(self, config: DualSystemClientConfig):
        self.config = config

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _request_timeout(self, path: str) -> float:
        route = path.strip("/").lower()
        if route == "dual_reset":
            reset_timeout = self.config.reset_timeout_sec
            if reset_timeout is not None and float(reset_timeout) > 0.0:
                return float(reset_timeout)
        return float(self.config.timeout_sec)

    def dual_reset(
        self,
        intrinsic: np.ndarray,
        instruction: str,
        *,
        navdp_url: str,
        s1_period_sec: float,
        s2_period_sec: float,
        goal_ttl_sec: float,
        traj_ttl_sec: float,
        traj_max_stale_sec: float,
    ) -> DualResetResponse:
        payload = {
            "intrinsic": np.asarray(intrinsic, dtype=np.float32).tolist(),
            "instruction": str(instruction),
            "navdp_url": str(navdp_url),
            "s1_period_sec": float(s1_period_sec),
            "s2_period_sec": float(s2_period_sec),
            "goal_ttl_sec": float(goal_ttl_sec),
            "traj_ttl_sec": float(traj_ttl_sec),
            "traj_max_stale_sec": float(traj_max_stale_sec),
        }
        last_error: Exception | None = None
        for _ in range(int(self.config.retry) + 1):
            try:
                resp = requests.post(
                    self._url("dual_reset"),
                    json=payload,
                    timeout=self._request_timeout("dual_reset"),
                )
                resp.raise_for_status()
                body = resp.json()
                algo = str(body.get("algo", ""))
                state = body.get("state")
                if algo == "":
                    raise DualSystemClientError("dual_reset returned empty algo")
                if not isinstance(state, dict):
                    raise DualSystemClientError("dual_reset returned invalid state")
                return DualResetResponse(algo=algo, state=state)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise DualSystemClientError(f"dual_reset failed after retries: {last_error}")

    def dual_step(
        self,
        *,
        rgb_image: np.ndarray,
        depth_image_m: np.ndarray,
        step_id: int,
        cam_pos: np.ndarray,
        cam_quat_wxyz: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
        events: dict[str, Any] | None = None,
    ) -> DualStepResponse:
        rgb = np.asarray(rgb_image, dtype=np.uint8)
        depth = np.asarray(depth_image_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb_image must be [H,W,3], got shape={rgb.shape}")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"depth_image_m must be [H,W] or [H,W,1], got shape={depth.shape}")

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        ok_img, img_buf = cv2.imencode(".jpg", rgb)
        if not ok_img:
            raise DualSystemClientError("failed to encode rgb_image as JPEG")
        depth_mm_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        ok_depth, depth_buf = cv2.imencode(".png", depth_mm_u16)
        if not ok_depth:
            raise DualSystemClientError("failed to encode depth_image_m as PNG")

        sensor_payload = sensor_meta if isinstance(sensor_meta, dict) else {}
        event_payload = events if isinstance(events, dict) else {}
        data = {
            "step_id": str(int(step_id)),
            "cam_pos": json.dumps(np.asarray(cam_pos, dtype=np.float32).reshape(-1).tolist()),
            "cam_quat_wxyz": json.dumps(np.asarray(cam_quat_wxyz, dtype=np.float32).reshape(-1).tolist()),
            "sensor_meta": json.dumps(sensor_payload),
            "events": json.dumps(event_payload),
        }
        files = {
            "image": ("image.jpg", io.BytesIO(img_buf).getvalue(), "image/jpeg"),
            "depth": ("depth.png", io.BytesIO(depth_buf).getvalue(), "image/png"),
        }

        last_error: Exception | None = None
        for _ in range(int(self.config.retry) + 1):
            try:
                resp = requests.post(
                    self._url("dual_step"),
                    files=files,
                    data=data,
                    timeout=self._request_timeout("dual_step"),
                )
                resp.raise_for_status()
                body = resp.json()
                traj = np.asarray(body.get("trajectory_world"), dtype=np.float32)
                if traj.ndim == 1 and traj.size == 0:
                    traj = np.zeros((0, 3), dtype=np.float32)
                if traj.ndim == 1:
                    traj = traj.reshape(-1, 3)
                if traj.ndim != 2:
                    raise DualSystemClientError(f"dual_step returned invalid trajectory shape: {traj.shape}")
                pixel_goal_raw = body.get("pixel_goal")
                pixel_goal = None
                if pixel_goal_raw is not None:
                    arr = np.asarray(pixel_goal_raw, dtype=np.float32).reshape(-1)
                    if arr.shape[0] >= 2:
                        pixel_goal = arr[:2].copy()
                debug = body.get("debug")
                if not isinstance(debug, dict):
                    debug = {}
                planner_control = body.get("planner_control")
                if not isinstance(planner_control, dict):
                    planner_control = {"mode": "trajectory", "yaw_delta_rad": None, "reason": ""}
                return DualStepResponse(
                    trajectory_world=traj,
                    pixel_goal=pixel_goal,
                    stop=bool(body.get("stop", False)),
                    goal_version=int(body.get("goal_version", -1)),
                    traj_version=int(body.get("traj_version", -1)),
                    used_cached_traj=bool(body.get("used_cached_traj", True)),
                    stale_sec=float(body.get("stale_sec", -1.0)),
                    planner_control=dict(planner_control),
                    debug=debug,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise DualSystemClientError(f"dual_step failed after retries: {last_error}")
