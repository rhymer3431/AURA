"""HTTP client for the external NavDP policy server."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
import time
from typing import Any

import numpy as np


def _require_dependencies():
    missing = []
    try:
        import requests
    except Exception:  # pragma: no cover - runtime dependency validation
        missing.append("requests")
        requests = None
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - runtime dependency validation
        missing.append("Pillow")
        Image = None
    if missing:
        raise RuntimeError(
            "NavDP integration requires "
            + ", ".join(missing)
            + " in the Isaac Sim Python environment."
        )
    return requests, Image


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3:
        raise ValueError(f"Expected an HxWxC RGB array, got shape {rgb.shape}.")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.max(initial=0.0) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected an HxW depth array, got shape {depth.shape}.")
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
    return depth_mm


def _image_bytes(rgb: np.ndarray, depth: np.ndarray) -> tuple[bytes, bytes]:
    _, Image = _require_dependencies()
    rgb_buffer = io.BytesIO()
    Image.fromarray(_normalize_rgb(rgb), mode="RGB").save(rgb_buffer, format="JPEG", quality=95)

    depth_buffer = io.BytesIO()
    Image.fromarray(_normalize_depth(depth), mode="I;16").save(depth_buffer, format="PNG")
    return rgb_buffer.getvalue(), depth_buffer.getvalue()


@dataclass(slots=True)
class NavDpPlan:
    """Single plan returned by the external NavDP policy."""

    trajectory_camera: np.ndarray
    all_trajectories_camera: np.ndarray | None
    values: np.ndarray | None
    plan_time_s: float
    stamp_s: float


class NavDpClient:
    """Typed adapter around the Flask server exposed by NavDP."""

    def __init__(
        self,
        server_url: str,
        timeout_s: float,
        *,
        fallback_mode: str = "disabled",
        heuristic_step_size_m: float = 0.35,
        heuristic_horizon: int = 8,
    ):
        self.server_url = str(server_url).rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        self.fallback_mode = str(fallback_mode).strip().lower()
        self.heuristic_step_size_m = max(1.0e-3, float(heuristic_step_size_m))
        self.heuristic_horizon = max(1, int(heuristic_horizon))
        self._active_backend = "http"
        self._status_message: str | None = None
        self._supports_pixelgoal: bool | None = None
        self._supports_imagegoal: bool | None = None

    @staticmethod
    def validate_runtime_dependencies():
        _require_dependencies()

    def consume_status_message(self) -> str | None:
        message = self._status_message
        self._status_message = None
        return message

    @property
    def supports_pixelgoal(self) -> bool | None:
        if self._active_backend == "heuristic":
            return False
        return self._supports_pixelgoal

    @property
    def supports_imagegoal(self) -> bool | None:
        if self._active_backend == "heuristic":
            return False
        return self._supports_imagegoal

    def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        requests, _ = _require_dependencies()
        response = requests.post(f"{self.server_url}{path}", timeout=self.timeout_s, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected a JSON object from NavDP, got {type(payload)!r}.")
        return payload

    def _switch_to_heuristic(self, exc: Exception) -> None:
        if self.fallback_mode != "heuristic" or self._active_backend == "heuristic":
            raise exc
        self._active_backend = "heuristic"
        self._supports_pixelgoal = False
        self._supports_imagegoal = False
        self._status_message = (
            f"NavDP server unavailable at {self.server_url}; "
            f"falling back to local heuristic planner: {type(exc).__name__}: {exc}"
        )

    def _heuristic_plan(self, point_goal_xy: np.ndarray) -> NavDpPlan:
        goal = np.asarray(point_goal_xy, dtype=np.float32).reshape(2)
        dist = float(np.linalg.norm(goal))
        if dist <= 1.0e-6:
            trajectory = np.zeros((1, 3), dtype=np.float32)
        else:
            steps = max(
                min(int(math.ceil(dist / self.heuristic_step_size_m)), self.heuristic_horizon),
                1,
            )
            alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
            trajectory = np.zeros((steps, 3), dtype=np.float32)
            trajectory[:, 0] = alphas * float(goal[0])
            trajectory[:, 1] = alphas * float(goal[1])

        return NavDpPlan(
            trajectory_camera=trajectory,
            all_trajectories_camera=trajectory[np.newaxis, ...],
            values=np.ones((1, trajectory.shape[0]), dtype=np.float32),
            plan_time_s=0.0,
            stamp_s=time.monotonic(),
        )

    def reset_pointgoal(self, intrinsic: np.ndarray, stop_threshold: float, batch_size: int = 1) -> str:
        if self._active_backend == "heuristic":
            self._supports_pixelgoal = False
            self._supports_imagegoal = False
            return "heuristic"
        try:
            payload = self._post(
                "/navigator_reset",
                json={
                    "intrinsic": np.asarray(intrinsic, dtype=np.float32).tolist(),
                    "stop_threshold": float(stop_threshold),
                    "batch_size": int(batch_size),
                },
            )
            algo = payload.get("algo")
            if not isinstance(algo, str):
                raise RuntimeError(f"NavDP reset response does not include a valid algorithm name: {payload}")
            self._supports_pixelgoal = bool(payload.get("supports_pixelgoal", False))
            self._supports_imagegoal = bool(payload.get("supports_imagegoal", False))
            return algo
        except Exception as exc:  # pragma: no cover - depends on runtime server availability
            self._switch_to_heuristic(exc)
            return "heuristic"

    def step_pointgoal(self, point_goal_xy: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> NavDpPlan:
        if self._active_backend == "heuristic":
            return self._heuristic_plan(point_goal_xy)
        goal = np.asarray(point_goal_xy, dtype=np.float32).reshape(2)
        try:
            rgb_bytes, depth_bytes = _image_bytes(rgb, depth)
            start = time.monotonic()
            payload = self._post(
                "/pointgoal_step",
                files={
                    "image": ("image.jpg", rgb_bytes, "image/jpeg"),
                    "depth": ("depth.png", depth_bytes, "image/png"),
                },
                data={
                    "goal_data": json.dumps({"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]}),
                    "depth_time": time.time(),
                    "rgb_time": time.time(),
                },
            )
            return NavDpPlan(
                trajectory_camera=np.asarray(payload.get("trajectory", []), dtype=np.float32).reshape(-1, 3),
                all_trajectories_camera=(
                    None
                    if payload.get("all_trajectory") is None
                    else np.asarray(payload.get("all_trajectory"), dtype=np.float32)
                ),
                values=(
                    None if payload.get("all_values") is None else np.asarray(payload.get("all_values"), dtype=np.float32)
                ),
                plan_time_s=float(time.monotonic() - start),
                stamp_s=time.monotonic(),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime server availability
            self._switch_to_heuristic(exc)
            return self._heuristic_plan(goal)

    def step_pixelgoal(self, pixel_goal_xy: np.ndarray, rgb: np.ndarray, depth: np.ndarray) -> NavDpPlan:
        if self._active_backend == "heuristic":
            raise RuntimeError("NavDP pixel-goal planning is unavailable in heuristic mode.")
        if self._supports_pixelgoal is False:
            raise RuntimeError("NavDP server does not support pixel-goal planning.")
        goal = np.asarray(pixel_goal_xy, dtype=np.float32).reshape(2)
        try:
            rgb_bytes, depth_bytes = _image_bytes(rgb, depth)
            start = time.monotonic()
            payload = self._post(
                "/pixelgoal_step",
                files={
                    "image": ("image.jpg", rgb_bytes, "image/jpeg"),
                    "depth": ("depth.png", depth_bytes, "image/png"),
                },
                data={
                    "goal_data": json.dumps({"goal_x": [int(round(float(goal[0])))], "goal_y": [int(round(float(goal[1])))]}),
                    "depth_time": time.time(),
                    "rgb_time": time.time(),
                },
            )
            return NavDpPlan(
                trajectory_camera=np.asarray(payload.get("trajectory", []), dtype=np.float32).reshape(-1, 3),
                all_trajectories_camera=(
                    None
                    if payload.get("all_trajectory") is None
                    else np.asarray(payload.get("all_trajectory"), dtype=np.float32)
                ),
                values=(
                    None if payload.get("all_values") is None else np.asarray(payload.get("all_values"), dtype=np.float32)
                ),
                plan_time_s=float(time.monotonic() - start),
                stamp_s=time.monotonic(),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime server availability
            if self.fallback_mode == "heuristic":
                self._switch_to_heuristic(exc)
            raise RuntimeError(f"NavDP pixel-goal request failed: {type(exc).__name__}: {exc}") from exc
