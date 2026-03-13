from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IceServerConfig:
    urls: tuple[str, ...]

    def as_public_dict(self) -> dict[str, object]:
        return {"urls": list(self.urls)}


def build_ice_server_configs(urls: Iterable[str]) -> tuple[IceServerConfig, ...]:
    configs: list[IceServerConfig] = []
    for raw_url in urls:
        normalized = str(raw_url).strip()
        if normalized == "":
            continue
        configs.append(IceServerConfig(urls=(normalized,)))
    return tuple(configs)


@dataclass(frozen=True)
class WebRTCGatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    control_endpoint: str = "tcp://127.0.0.1:5580"
    telemetry_endpoint: str = "tcp://127.0.0.1:5581"
    shm_name: str = "g1_view_frames"
    shm_slot_size: int = 8 * 1024 * 1024
    shm_capacity: int = 8
    enable_depth_track: bool = True
    rgb_fps: float = 15.0
    depth_fps: float = 5.0
    telemetry_hz: float = 10.0
    state_snapshot_hz: float = 1.0
    poll_interval_ms: int = 20
    stale_frame_timeout_sec: float = 2.0
    cors_origin: str = "*"
    identity: str = "webrtc_gateway"
    observe_only: bool = True
    peer_model: str = "single"
    channel_labels: tuple[str, str] = ("state", "telemetry")
    ice_servers: tuple[IceServerConfig, ...] = ()

    def public_config(self) -> dict[str, object]:
        return {
            "observeOnly": bool(self.observe_only),
            "peerModel": str(self.peer_model),
            "channelLabels": list(self.channel_labels),
            "iceServers": [item.as_public_dict() for item in self.ice_servers],
            "enableDepthTrack": bool(self.enable_depth_track),
        }
