from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.cv2_compat import cv2

from .base import NavDPNoGoalResponse, NavDPPointGoalResponse


@dataclass(frozen=True)
class NavDPExecutorConfig:
    checkpoint_path: str
    device: str = "cpu"
    amp: bool = False
    amp_dtype: str = "float16"
    tf32: bool = False
    stop_threshold: float = -3.0
    image_size: int = 224


class NavDPExecutorBackend:
    backend_name = "unknown"

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        raise NotImplementedError

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        raise NotImplementedError

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        raise NotImplementedError


class HeuristicNavDPExecutor(NavDPExecutorBackend):
    backend_name = "heuristic"

    def __init__(self, *, step_size_m: float = 0.35, horizon: int = 8) -> None:
        self._step_size_m = float(step_size_m)
        self._horizon = int(horizon)
        self._batch_size = 1

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        _ = intrinsic
        self._batch_size = int(batch_size)
        return self.backend_name

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        _ = rgb_images, depth_images_m, sensor_meta
        goals = np.asarray(point_goals, dtype=np.float32)
        trajectories = [self._line_trajectory(goal[:2]) for goal in goals]
        all_trajectory = np.asarray(trajectories, dtype=np.float32)
        all_values = np.ones((all_trajectory.shape[0], all_trajectory.shape[1]), dtype=np.float32)
        return NavDPPointGoalResponse(
            trajectory=all_trajectory[0],
            all_trajectory=all_trajectory,
            all_values=all_values,
            server_input_meta={"backend": self.backend_name},
        )

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        _ = rgb_images, depth_images_m
        trajectory = self._line_trajectory(np.asarray([self._step_size_m * self._horizon, 0.0], dtype=np.float32))
        all_trajectory = trajectory.reshape(1, trajectory.shape[0], trajectory.shape[1])
        all_values = np.ones((1, trajectory.shape[0]), dtype=np.float32)
        return NavDPNoGoalResponse(trajectory=trajectory, all_trajectory=all_trajectory, all_values=all_values)

    def _line_trajectory(self, goal_xy: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        dist = float(np.linalg.norm(goal[:2]))
        if dist <= 1.0e-6:
            return np.zeros((1, 3), dtype=np.float32)
        steps = max(min(int(math.ceil(dist / max(self._step_size_m, 1.0e-3))), self._horizon), 1)
        alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
        traj = np.zeros((steps, 3), dtype=np.float32)
        traj[:, 0] = alphas * float(goal[0])
        traj[:, 1] = alphas * float(goal[1])
        return traj


class PolicyNavDPExecutor(NavDPExecutorBackend):
    backend_name = "policy"

    def __init__(self, config: NavDPExecutorConfig) -> None:
        self.config = config
        self._navigator = None
        self._intrinsic: np.ndarray | None = None
        self._batch_size = 1

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        self._batch_size = int(batch_size)
        self._intrinsic = np.asarray(intrinsic, dtype=np.float32)
        self._ensure_navigator()
        assert self._navigator is not None
        self._navigator.reset(self._batch_size, np.asarray(self.config.stop_threshold))
        return self.backend_name

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        _ = sensor_meta
        navigator = self._ensure_navigator()
        goals = np.asarray(point_goals, dtype=np.float32)
        if goals.ndim != 2:
            raise ValueError(f"point_goals must be rank-2, got {goals.shape}")
        goal_xyz = np.concatenate([goals[:, :2], np.zeros((goals.shape[0], 1), dtype=np.float32)], axis=1)
        rgb = self._prepare_rgb(rgb_images)
        depth = self._prepare_depth(depth_images_m)
        trajectory, all_trajectory, all_values = navigator.step_pointgoal(goal_xyz, rgb, depth)
        return NavDPPointGoalResponse(
            trajectory=np.asarray(trajectory[0], dtype=np.float32),
            all_trajectory=np.asarray(all_trajectory, dtype=np.float32),
            all_values=np.asarray(all_values, dtype=np.float32),
            server_input_meta={"backend": self.backend_name},
        )

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        navigator = self._ensure_navigator()
        rgb = self._prepare_rgb(rgb_images)
        depth = self._prepare_depth(depth_images_m)
        trajectory, all_trajectory, all_values = navigator.step_nogoal(rgb, depth)
        return NavDPNoGoalResponse(
            trajectory=np.asarray(trajectory[0], dtype=np.float32),
            all_trajectory=np.asarray(all_trajectory, dtype=np.float32),
            all_values=np.asarray(all_values, dtype=np.float32),
        )

    def _ensure_navigator(self):
        if self._navigator is not None:
            return self._navigator
        if self._intrinsic is None:
            raise RuntimeError("navigator_reset must be called before planner steps")
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint_path}")
        from inference.policy_agent import NavDP_Agent

        self._navigator = NavDP_Agent(
            self._intrinsic,
            image_size=int(self.config.image_size),
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=str(checkpoint_path),
            device=str(self.config.device),
            use_amp=bool(self.config.amp),
            amp_dtype=str(self.config.amp_dtype),
            enable_tf32=bool(self.config.tf32),
        )
        return self._navigator

    @staticmethod
    def _prepare_rgb(rgb_images: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb_images, dtype=np.uint8)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb_images must be [B,H,W,3], got {rgb.shape}")
        bgr = np.empty_like(rgb)
        for index in range(rgb.shape[0]):
            bgr[index] = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
        return bgr

    @staticmethod
    def _prepare_depth(depth_images_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_images_m, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., np.newaxis]
        if depth.ndim != 4:
            raise ValueError(f"depth_images_m must be [B,H,W] or [B,H,W,1], got {depth.shape}")
        return depth
