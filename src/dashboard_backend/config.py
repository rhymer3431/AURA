from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from webrtc.config import IceServerConfig


@dataclass(frozen=True)
class DashboardBackendConfig:
    host: str = "127.0.0.1"
    port: int = 8095
    repo_root: Path = Path(".")
    dashboard_dir: Path = Path("dashboard")
    dev_origin: str = "http://127.0.0.1:5173"
    control_endpoint: str = "tcp://127.0.0.1:5580"
    telemetry_endpoint: str = "tcp://127.0.0.1:5581"
    shm_name: str = "g1_view_frames"
    shm_slot_size: int = 8 * 1024 * 1024
    shm_capacity: int = 8
    enable_depth_track: bool = True
    rgb_fps: float = 15.0
    depth_fps: float = 5.0
    telemetry_hz: float = 10.0
    ice_servers: tuple[IceServerConfig, ...] = ()
    health_poll_interval_sec: float = 1.0

    @property
    def dist_dir(self) -> Path:
        return self.dashboard_dir / "dist"

    @property
    def process_log_dir(self) -> Path:
        return self.repo_root / "tmp" / "process_logs" / "dashboard"
