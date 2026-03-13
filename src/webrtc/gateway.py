from __future__ import annotations

from typing import Any

from .config import WebRTCGatewayConfig, cors_header_for_origin
from .session import PeerSessionManager
from .subscriber import ObservationSubscriber


class WebRTCGateway:
    def __init__(
        self,
        config: WebRTCGatewayConfig,
        *,
        subscriber: ObservationSubscriber | None = None,
        session_manager: PeerSessionManager | None = None,
    ) -> None:
        try:
            from aiohttp import web
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("aiohttp is required for the WebRTC gateway app.") from exc

        self._web = web
        self.config = config
        self.subscriber = subscriber or ObservationSubscriber(config)
        self.session_manager = session_manager or PeerSessionManager(config, self.subscriber)

    def create_app(self):
        app = self._web.Application(middlewares=[self._cors_middleware])
        app["webrtc_gateway"] = self
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/config", self.handle_config)
        app.router.add_post("/offer", self.handle_offer)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app) -> None:  # noqa: ANN001
        _ = app
        await self.subscriber.start()

    async def _on_cleanup(self, app) -> None:  # noqa: ANN001
        _ = app
        await self.session_manager.close()
        await self.subscriber.close()

    @property
    def _cors_middleware(self):
        @self._web.middleware
        async def middleware(request, handler):  # noqa: ANN001
            if request.method == "OPTIONS":
                response = self._web.Response(status=200)
            else:
                response = await handler(request)
            origin_header = cors_header_for_origin(request.headers.get("Origin", ""), self.config.cors_origins)
            if origin_header is not None:
                response.headers["Access-Control-Allow-Origin"] = origin_header
                if origin_header != "*":
                    response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        return middleware

    async def handle_health(self, request):  # noqa: ANN001
        _ = request
        session = self.session_manager.active_session
        frame = self.subscriber.current_frame
        return self._web.json_response(
            {
                "status": "ok",
                "service": "webrtc_gateway",
                "peer": {
                    "active": session is not None,
                    "sessionId": None if session is None else session.session_id,
                    "trackRoles": [] if session is None else list(session.track_roles),
                },
                "frame": {
                    "available": frame is not None,
                    "ageMs": self.subscriber.last_frame_age_ms(),
                    "seq": None if frame is None else int(frame.seq),
                },
                "transport": {
                    "controlEndpoint": str(self.config.control_endpoint),
                    "telemetryEndpoint": str(self.config.telemetry_endpoint),
                    "shmName": str(self.config.shm_name),
                },
            }
        )

    async def handle_config(self, request):  # noqa: ANN001
        _ = request
        return self._web.json_response(self.config.public_config())

    async def handle_offer(self, request):  # noqa: ANN001
        payload = await request.json()
        if not isinstance(payload, dict):
            raise self._web.HTTPBadRequest(reason="offer payload must be a JSON object")
        if str(payload.get("type", "")).strip().lower() != "offer":
            raise self._web.HTTPBadRequest(reason="offer payload must have type=offer")
        try:
            session, answer = await self.session_manager.accept_offer(payload)
        except RuntimeError as exc:
            raise self._web.HTTPBadRequest(reason=str(exc)) from exc
        return self._web.json_response(
            {
                "sdp": str(answer.sdp),
                "type": str(answer.type),
                "sessionId": str(session.session_id),
            }
        )
