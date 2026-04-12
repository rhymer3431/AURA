from __future__ import annotations

from backend.webrtc.config import WebRTCServiceConfig


def test_webrtc_defaults_prioritize_direct_rgb_streaming() -> None:
    config = WebRTCServiceConfig()

    assert config.enable_depth_track is False
    assert config.rgb_fps == 30.0
    assert config.depth_fps == 15.0
    assert config.telemetry_hz == 15.0
    assert config.poll_interval_ms == 10

    public = config.public_config(enabled=True)
    assert public["transportMode"] == "webrtc"
    assert public["enableDepthTrack"] is False
    assert public["rgbFps"] == 30.0
    assert public["depthFps"] == 0.0
    assert public["telemetryHz"] == 15.0
