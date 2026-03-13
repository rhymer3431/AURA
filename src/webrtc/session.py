from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import uuid

from .config import WebRTCGatewayConfig
from .models import build_session_ready_message
from .subscriber import ObservationSubscriber
from .tracks import DepthPreviewVideoTrack, RgbVideoTrack

try:
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
except Exception as exc:  # noqa: BLE001
    RTCConfiguration = None  # type: ignore[assignment]
    RTCIceServer = None  # type: ignore[assignment]
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    _RTC_IMPORT_ERROR = exc
else:
    _RTC_IMPORT_ERROR = None


def _require_rtc_dependencies() -> None:
    global RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, _RTC_IMPORT_ERROR
    if _RTC_IMPORT_ERROR is not None:
        try:
            from aiortc import RTCConfiguration as _RTCConfiguration, RTCIceServer as _RTCIceServer, RTCPeerConnection as _RTCPeerConnection, RTCSessionDescription as _RTCSessionDescription
        except Exception:
            pass
        else:
            RTCConfiguration = _RTCConfiguration
            RTCIceServer = _RTCIceServer
            RTCPeerConnection = _RTCPeerConnection
            RTCSessionDescription = _RTCSessionDescription
            _RTC_IMPORT_ERROR = None
    if _RTC_IMPORT_ERROR is not None:
        raise RuntimeError("aiortc is required for WebRTC peer sessions.") from _RTC_IMPORT_ERROR


async def wait_for_ice_gathering_complete(peer_connection, *, timeout_sec: float = 10.0) -> None:  # noqa: ANN001
    if peer_connection.iceGatheringState == "complete":
        return
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    @peer_connection.on("icegatheringstatechange")
    def _on_ice_gathering_state_change() -> None:
        if peer_connection.iceGatheringState == "complete" and not waiter.done():
            waiter.set_result(None)

    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(waiter, timeout=max(float(timeout_sec), 0.1))


class WebRTCPeerSession:
    def __init__(self, session_id: str, config: WebRTCGatewayConfig, subscriber: ObservationSubscriber) -> None:
        _require_rtc_dependencies()
        self.session_id = str(session_id)
        self.config = config
        self.subscriber = subscriber
        self.track_roles = ["rgb"]
        self._listener = subscriber.add_listener()
        self._state_channel = None
        self._telemetry_channel = None
        self._closed = False
        self._pc = RTCPeerConnection(configuration=self._build_rtc_configuration(config))
        self._pc.addTrack(RgbVideoTrack(subscriber, fps=config.rgb_fps))
        current = subscriber.current_frame
        if bool(config.enable_depth_track) and current is not None and current.depth_image_m is not None:
            self._pc.addTrack(DepthPreviewVideoTrack(subscriber, fps=config.depth_fps))
            self.track_roles.append("depth")
        self._tasks = [
            asyncio.create_task(self._state_event_loop(), name=f"webrtc-state-events-{self.session_id}"),
            asyncio.create_task(self._state_snapshot_loop(), name=f"webrtc-state-snapshot-{self.session_id}"),
            asyncio.create_task(self._telemetry_loop(), name=f"webrtc-telemetry-{self.session_id}"),
        ]

        @self._pc.on("datachannel")
        def _on_datachannel(channel) -> None:  # noqa: ANN001
            self._configure_data_channel(channel)

        @self._pc.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            if self._pc.connectionState in {"closed", "failed"}:
                await self.close()

    async def accept_offer(self, offer_payload: dict[str, object]):
        description = RTCSessionDescription(
            sdp=str(offer_payload.get("sdp", "")),
            type=str(offer_payload.get("type", "offer")),
        )
        await self._pc.setRemoteDescription(description)
        try:
            await self._pc.setLocalDescription(await self._pc.createAnswer())
        except ValueError as exc:
            raise RuntimeError(
                "offer must include recvonly video transceivers for the gateway video tracks"
            ) from exc
        await wait_for_ice_gathering_complete(self._pc)
        assert self._pc.localDescription is not None
        return self._pc.localDescription

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.subscriber.remove_listener(self._listener)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        await self._pc.close()

    def _configure_data_channel(self, channel) -> None:  # noqa: ANN001
        label = str(getattr(channel, "label", "")).strip().lower()
        if label == self.config.channel_labels[0]:
            self._state_channel = channel

            @channel.on("open")
            def _on_open() -> None:
                asyncio.create_task(self._send_state_ready(), name=f"webrtc-state-ready-{self.session_id}")

            @channel.on("close")
            def _on_close() -> None:
                self._state_channel = None

            if str(getattr(channel, "readyState", "")).lower() == "open":
                asyncio.create_task(self._send_state_ready(), name=f"webrtc-state-ready-{self.session_id}")

        if label == self.config.channel_labels[1]:
            self._telemetry_channel = channel

            @channel.on("open")
            def _on_open() -> None:
                asyncio.create_task(self._send_initial_frame_meta(), name=f"webrtc-telemetry-ready-{self.session_id}")

            @channel.on("close")
            def _on_close() -> None:
                self._telemetry_channel = None

            if str(getattr(channel, "readyState", "")).lower() == "open":
                asyncio.create_task(self._send_initial_frame_meta(), name=f"webrtc-telemetry-ready-{self.session_id}")

    async def _send_state_ready(self) -> None:
        await self._send_json(
            self._state_channel,
            build_session_ready_message(
                session_id=self.session_id,
                track_roles=self.track_roles,
                channel_labels=self.config.channel_labels,
            ),
        )
        await self._send_json(self._state_channel, self.subscriber.build_state_snapshot())

    async def _send_initial_frame_meta(self) -> None:
        payload = self.subscriber.build_frame_meta()
        if payload is not None:
            await self._send_json(self._telemetry_channel, payload)

    async def _state_event_loop(self) -> None:
        while True:
            event = await self._listener.get()
            await self._send_json(self._state_channel, event.payload)

    async def _state_snapshot_loop(self) -> None:
        period = 1.0 / max(float(self.config.state_snapshot_hz), 0.1)
        while True:
            await asyncio.sleep(period)
            await self._send_json(self._state_channel, self.subscriber.build_state_snapshot())

    async def _telemetry_loop(self) -> None:
        period = 1.0 / max(float(self.config.telemetry_hz), 0.1)
        last_seq = -1
        while True:
            await asyncio.sleep(period)
            payload = self.subscriber.build_frame_meta()
            if payload is None:
                continue
            seq = int(payload.get("seq", -1))
            if seq == last_seq:
                continue
            await self._send_json(self._telemetry_channel, payload)
            last_seq = seq

    async def _send_json(self, channel, payload: dict[str, object] | None) -> None:  # noqa: ANN001
        if payload is None or channel is None:
            return
        if str(getattr(channel, "readyState", "")).lower() != "open":
            return
        try:
            channel.send(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        except Exception:
            return

    @staticmethod
    def _build_rtc_configuration(config: WebRTCGatewayConfig):
        servers = [RTCIceServer(urls=list(item.urls)) for item in config.ice_servers]
        return RTCConfiguration(iceServers=servers)


class PeerSessionManager:
    def __init__(
        self,
        config: WebRTCGatewayConfig,
        subscriber: ObservationSubscriber,
        *,
        session_factory=None,
    ) -> None:
        self.config = config
        self.subscriber = subscriber
        self._session_factory = session_factory or WebRTCPeerSession
        self._current = None
        self._lock = asyncio.Lock()

    @property
    def active_session(self):
        return self._current

    async def accept_offer(self, offer_payload: dict[str, object]):
        async with self._lock:
            session_id = uuid.uuid4().hex[:12]
            session = self._session_factory(session_id, self.config, self.subscriber)
            previous = self._current
            self._current = session
            if previous is not None:
                await previous.close()
            try:
                answer = await session.accept_offer(offer_payload)
            except Exception:
                await session.close()
                if self._current is session:
                    self._current = None
                raise
            return session, answer

    async def close(self) -> None:
        async with self._lock:
            current = self._current
            self._current = None
        if current is not None:
            await current.close()
