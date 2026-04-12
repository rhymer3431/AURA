"""Configuration primitives for the backend-owned WebRTC viewer."""

from __future__ import annotations

from dataclasses import dataclass

from systems.shared.viewer_transport import (
    VIEWER_CONTROL_ENDPOINT,
    VIEWER_SHM_CAPACITY,
    VIEWER_SHM_NAME,
    VIEWER_SHM_SLOT_SIZE,
    VIEWER_TELEMETRY_ENDPOINT,
)


@dataclass(frozen=True)
class IceServerConfig:
    urls: tuple[str, ...]

    def as_public_dict(self) -> dict[str, object]:
        return {"urls": list(self.urls)}


@dataclass(frozen=True)
class WebRTCServiceConfig:
    control_endpoint: str = VIEWER_CONTROL_ENDPOINT
    telemetry_endpoint: str = VIEWER_TELEMETRY_ENDPOINT
    shm_name: str = VIEWER_SHM_NAME
    shm_slot_size: int = VIEWER_SHM_SLOT_SIZE
    shm_capacity: int = VIEWER_SHM_CAPACITY
    enable_depth_track: bool = False
    rgb_fps: float = 30.0
    depth_fps: float = 15.0
    telemetry_hz: float = 15.0
    state_snapshot_hz: float = 2.0
    poll_interval_ms: int = 10
    stale_frame_timeout_sec: float = 2.0
    identity: str = "backend_webrtc"
    observe_only: bool = True
    peer_model: str = "single"
    channel_labels: tuple[str, str] = ("state", "telemetry")
    ice_servers: tuple[IceServerConfig, ...] = ()

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        return {
            "transportMode": "webrtc" if enabled else "disabled",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc" if enabled else "disabled",
            "observeOnly": bool(self.observe_only),
            "peerModel": str(self.peer_model),
            "channelLabels": list(self.channel_labels),
            "iceServers": [item.as_public_dict() for item in self.ice_servers],
            "enableDepthTrack": bool(self.enable_depth_track),
            "rgbFps": float(self.rgb_fps),
            "depthFps": float(self.depth_fps) if self.enable_depth_track else 0.0,
            "telemetryHz": float(self.telemetry_hz),
            "proxyMode": "internal",
        }
