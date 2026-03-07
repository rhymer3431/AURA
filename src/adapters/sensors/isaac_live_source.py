from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .d455_sensor import D455SensorAdapter, D455SensorAdapterConfig
from .frame_source import FrameSample, FrameSourceReport


@dataclass(frozen=True)
class IsaacLiveSourceConfig:
    source_name: str = "isaac_live"
    strict_live: bool = False
    image_width: int = 640
    image_height: int = 640
    depth_max_m: float = 5.0
    force_runtime_mount: bool = False


class IsaacLiveFrameSource:
    def __init__(
        self,
        *,
        simulation_app=None,
        stage=None,
        env_provider: Callable[[], object] | None = None,
        robot_pose_provider: Callable[[], tuple[float, float, float]] | None = None,
        robot_yaw_provider: Callable[[], float] | None = None,
        sensor_factory: Callable[[D455SensorAdapterConfig], D455SensorAdapter] | None = None,
        sensor_adapter: D455SensorAdapter | None = None,
        config: IsaacLiveSourceConfig | None = None,
    ) -> None:
        self._simulation_app = simulation_app
        self._stage = stage
        self._env_provider = env_provider
        self._robot_pose_provider = robot_pose_provider or (lambda: (0.0, 0.0, 0.0))
        self._robot_yaw_provider = robot_yaw_provider or (lambda: 0.0)
        self._config = config or IsaacLiveSourceConfig()
        self._provided_sensor_adapter = sensor_adapter is not None
        cfg = D455SensorAdapterConfig(
            use_d455=True,
            image_width=int(self._config.image_width),
            image_height=int(self._config.image_height),
            depth_max_m=float(self._config.depth_max_m),
            strict_d455=bool(self._config.strict_live),
            force_runtime_mount=bool(self._config.force_runtime_mount),
        )
        self._sensor = sensor_adapter or (sensor_factory or D455SensorAdapter)(cfg)
        self._started = False
        self._frame_id = 0
        self._report = FrameSourceReport(source_name=self._config.source_name, status="pending")

    @property
    def sensor(self) -> D455SensorAdapter:
        return self._sensor

    def start(self) -> FrameSourceReport:
        if self._started:
            return self._report
        if self._provided_sensor_adapter and (self._simulation_app is None or self._stage is None):
            self._started = True
            self._report = FrameSourceReport(
                source_name=self._config.source_name,
                status="ready",
                live_available=True,
                fallback_used=False,
                notice="Reusing initialized Isaac sensor adapter.",
                details={"capture_report": self._sensor.last_capture_meta},
            )
            return self._report
        if self._simulation_app is None or self._stage is None:
            self._report = FrameSourceReport(
                source_name=self._config.source_name,
                status="unavailable",
                live_available=False,
                fallback_used=False,
                notice="Isaac live source requires simulation_app and stage.",
            )
            return self._report
        ok, message = self._sensor.initialize(self._simulation_app, self._stage)
        self._started = bool(ok)
        self._report = FrameSourceReport(
            source_name=self._config.source_name,
            status="ready" if ok else "unavailable",
            live_available=bool(ok),
            fallback_used=not bool(ok),
            notice=str(message),
            details={"capture_report": self._sensor.last_capture_meta},
        )
        return self._report

    def read(self) -> FrameSample | None:
        if not self._started:
            report = self.start()
            if report.status != "ready":
                return None
        env = self._env_provider() if callable(self._env_provider) else None
        rgb, depth, capture_meta = self._sensor.capture_rgbd_with_meta(env)
        if rgb is None or depth is None:
            self._report = FrameSourceReport(
                source_name=self._config.source_name,
                status="degraded" if rgb is not None or depth is not None else "unavailable",
                live_available=False,
                fallback_used=True,
                notice=str(capture_meta.get("note", "live frame capture failed")),
                details={"capture_report": capture_meta},
            )
            return None
        cam_pos, cam_quat = self._sensor.get_rgb_camera_pose_world()
        self._frame_id += 1
        robot_pose = tuple(float(v) for v in self._robot_pose_provider())
        robot_yaw = float(self._robot_yaw_provider())
        sim_time_s = self._extract_sim_time(env)
        self._report = FrameSourceReport(
            source_name=self._config.source_name,
            status="ready",
            live_available=True,
            fallback_used=bool(capture_meta.get("fallback_used", False)),
            notice=str(capture_meta.get("note", "")),
            details={"capture_report": capture_meta},
        )
        return FrameSample(
            frame_id=self._frame_id,
            source_name=self._config.source_name,
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            camera_pose_xyz=tuple(float(v) for v in (cam_pos if cam_pos is not None else np.zeros(3, dtype=np.float32))[:3]),
            camera_quat_wxyz=tuple(
                float(v)
                for v in (cam_quat if cam_quat is not None else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))[:4]
            ),
            robot_pose_xyz=robot_pose,
            robot_yaw_rad=robot_yaw,
            camera_intrinsic=self._sensor.intrinsic,
            sim_time_s=sim_time_s,
            metadata={
                "frame_source": self._config.source_name,
                "capture_report": capture_meta,
            },
        )

    def close(self) -> None:
        self._started = False

    def report(self) -> FrameSourceReport:
        return self._report

    @staticmethod
    def _extract_sim_time(env) -> float:  # noqa: ANN001
        if env is None:
            return float(time.time())
        for attr_name in ("sim_time", "current_time", "time"):
            value = getattr(env, attr_name, None)
            if isinstance(value, (int, float)):
                return float(value)
        try:
            data = getattr(getattr(env, "unwrapped", env), "sim", None)
            if data is not None and isinstance(getattr(data, "current_time", None), (int, float)):
                return float(data.current_time)
        except Exception:  # noqa: BLE001
            return float(time.time())
        return float(time.time())
