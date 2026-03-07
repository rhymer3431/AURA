from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import NavDPExecutionClient, NavDPNoGoalResponse, NavDPPointGoalResponse
from .executor import HeuristicNavDPExecutor, NavDPExecutorBackend, NavDPExecutorConfig, PolicyNavDPExecutor


@dataclass(frozen=True)
class InProcessNavDPClientConfig:
    backend: str = "auto"
    checkpoint_path: str = ""
    device: str = "cpu"
    amp: bool = False
    amp_dtype: str = "float16"
    tf32: bool = False
    stop_threshold: float = -3.0


class InProcessNavDPClient(NavDPExecutionClient):
    def __init__(self, config: InProcessNavDPClientConfig, *, backend_impl: NavDPExecutorBackend | None = None) -> None:
        self.config = config
        self._auto_backend = str(config.backend).strip().lower() == "auto"
        self.backend_impl = backend_impl or self._build_backend()
        self._last_intrinsic: np.ndarray | None = None
        self._last_batch_size = 1

    @property
    def backend_name(self) -> str:
        return str(getattr(self.backend_impl, "backend_name", "unknown"))

    def navigator_reset(self, intrinsic: np.ndarray, batch_size: int = 1) -> str:
        intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
        self._last_intrinsic = intrinsic_array
        self._last_batch_size = int(batch_size)
        try:
            return self.backend_impl.navigator_reset(intrinsic_array, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            self._fallback_backend(exc)
            return self.backend_impl.navigator_reset(intrinsic_array, batch_size=batch_size)

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
        sensor_meta: dict[str, Any] | None = None,
    ) -> NavDPPointGoalResponse:
        try:
            return self.backend_impl.pointgoal_step(point_goals, rgb_images, depth_images_m, sensor_meta=sensor_meta)
        except Exception as exc:  # noqa: BLE001
            self._fallback_backend(exc)
            return self.backend_impl.pointgoal_step(point_goals, rgb_images, depth_images_m, sensor_meta=sensor_meta)

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images_m: np.ndarray,
    ) -> NavDPNoGoalResponse:
        try:
            return self.backend_impl.nogoal_step(rgb_images, depth_images_m)
        except Exception as exc:  # noqa: BLE001
            self._fallback_backend(exc)
            return self.backend_impl.nogoal_step(rgb_images, depth_images_m)

    def _build_backend(self) -> NavDPExecutorBackend:
        backend = str(self.config.backend).strip().lower()
        if backend == "heuristic":
            return HeuristicNavDPExecutor()
        if backend == "policy":
            return PolicyNavDPExecutor(
                NavDPExecutorConfig(
                    checkpoint_path=self.config.checkpoint_path,
                    device=self.config.device,
                    amp=self.config.amp,
                    amp_dtype=self.config.amp_dtype,
                    tf32=self.config.tf32,
                    stop_threshold=self.config.stop_threshold,
                )
            )
        if backend != "auto":
            raise ValueError(f"Unsupported in-process NavDP backend: {self.config.backend}")

        try:
            return PolicyNavDPExecutor(
                NavDPExecutorConfig(
                    checkpoint_path=self.config.checkpoint_path,
                    device=self.config.device,
                    amp=self.config.amp,
                    amp_dtype=self.config.amp_dtype,
                    tf32=self.config.tf32,
                    stop_threshold=self.config.stop_threshold,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back to heuristic NavDP executor: {type(exc).__name__}: {exc}", stacklevel=2)
            return HeuristicNavDPExecutor()

    def _fallback_backend(self, exc: Exception) -> None:
        if not self._auto_backend or isinstance(self.backend_impl, HeuristicNavDPExecutor):
            raise exc
        warnings.warn(f"Falling back to heuristic NavDP executor: {type(exc).__name__}: {exc}", stacklevel=2)
        self.backend_impl = HeuristicNavDPExecutor()
        if self._last_intrinsic is not None:
            self.backend_impl.navigator_reset(self._last_intrinsic, batch_size=self._last_batch_size)
