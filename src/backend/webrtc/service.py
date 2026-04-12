"""Backend-owned WebRTC gateway service."""

from __future__ import annotations

from typing import Any

from .config import WebRTCServiceConfig
from .session import PeerSessionManager
from .subscriber import ObservationSubscriber


class WebRTCService:
    def __init__(
        self,
        config: WebRTCServiceConfig | None = None,
        *,
        subscriber: ObservationSubscriber | None = None,
        session_manager: PeerSessionManager | None = None,
    ) -> None:
        self.config = config or WebRTCServiceConfig()
        self.subscriber = subscriber or ObservationSubscriber(self.config)
        self.session_manager = session_manager or PeerSessionManager(self.config, self.subscriber)

    async def start(self) -> None:
        await self.subscriber.start()

    async def close(self) -> None:
        await self.session_manager.close()
        await self.subscriber.close()

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        return self.config.public_config(enabled=enabled)

    async def accept_offer(self, offer_payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(offer_payload, dict):
            raise RuntimeError("offer payload must be a JSON object")
        if str(offer_payload.get("type", "")).strip().lower() != "offer":
            raise RuntimeError("offer payload must have type=offer")
        session, answer = await self.session_manager.accept_offer(offer_payload)
        return {
            "sdp": str(answer.sdp),
            "type": str(answer.type),
            "sessionId": str(session.session_id),
        }

    def health_snapshot(self) -> dict[str, Any]:
        session = self.session_manager.active_session
        frame = self.subscriber.current_frame
        frame_age = self.subscriber.last_frame_age_ms()
        frame_available = self.subscriber.has_fresh_frame()
        stream_stalled = bool(frame is not None and not frame_available)
        drop_counters = {
            "shmOverwrite": int(self.subscriber.shm_overwrite_drops()),
        }
        debug_counters = self.subscriber.debug_counters
        return {
            "transport": "webrtc",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc",
            "rgbFps": float(self.config.rgb_fps),
            "depthFps": float(self.config.depth_fps) if self.config.enable_depth_track else 0.0,
            "enableDepthTrack": bool(self.config.enable_depth_track),
            "frameAvailable": bool(frame_available),
            "streamStalled": stream_stalled,
            "frameSeq": None if frame is None else int(frame.seq),
            "frameId": None if frame is None else int(frame.frame_header.frame_id),
            "frameAgeMs": frame_age,
            "lastGoodFrameAgeMs": frame_age,
            "peerActive": session is not None,
            "peerSessionId": None if session is None else str(session.session_id),
            "peerTrackRoles": [] if session is None else list(session.track_roles),
            "rgbAvailable": bool(frame_available),
            "depthAvailable": bool(frame_available and frame is not None and frame.depth_image_m is not None),
            "source": "control_runtime" if frame is None else str(frame.frame_header.source),
            "image": {
                "width": 0 if frame is None else int(frame.frame_header.width),
                "height": 0 if frame is None else int(frame.frame_header.height),
            },
            "dropCounters": drop_counters,
            "transportHealth": {
                "control_endpoint": str(self.config.control_endpoint),
                "telemetry_endpoint": str(self.config.telemetry_endpoint),
                "shm_name": str(self.config.shm_name),
                "decodeOk": int(debug_counters.get("decodeOk", 0)),
                "decodeDrops": int(debug_counters.get("decodeDrops", 0)),
                "shmOverwriteDrops": int(debug_counters.get("shmOverwriteDrops", 0)),
                "staleTransitions": int(debug_counters.get("staleTransitions", 0)),
            },
            "latestHealth": self.subscriber.latest_health,
        }
