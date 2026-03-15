from __future__ import annotations

import asyncio
from contextlib import suppress
import json

from aiohttp import web

from webrtc.config import WebRTCGatewayConfig, cors_header_for_origin
from webrtc.gateway import WebRTCGateway
from webrtc.session import PeerSessionManager
from webrtc.subscriber import ObservationSubscriber

from .config import DashboardBackendConfig
from .log_tailer import LogTailer
from .models import parse_session_request
from .process_manager import ProcessManager
from .runtime_control import RuntimeControlClient
from .state import StateAggregator


class DashboardWebApp:
    def __init__(
        self,
        config: DashboardBackendConfig,
        *,
        process_manager: ProcessManager | None = None,
        control_client: RuntimeControlClient | None = None,
        subscriber: ObservationSubscriber | None = None,
        session_manager: PeerSessionManager | None = None,
        log_tailer: LogTailer | None = None,
        state_aggregator: StateAggregator | None = None,
    ) -> None:
        self.config = config
        self.process_manager = process_manager or ProcessManager(config)
        self.control_client = control_client or RuntimeControlClient(
            control_endpoint=config.control_endpoint,
            telemetry_endpoint=config.telemetry_endpoint,
        )
        self.subscriber = subscriber or ObservationSubscriber(
            WebRTCGatewayConfig(
                control_endpoint=config.control_endpoint,
                telemetry_endpoint=config.telemetry_endpoint,
                shm_name=config.shm_name,
                shm_slot_size=config.shm_slot_size,
                shm_capacity=config.shm_capacity,
                enable_depth_track=config.enable_depth_track,
                rgb_fps=config.rgb_fps,
                depth_fps=config.depth_fps,
                telemetry_hz=config.telemetry_hz,
                ice_servers=config.ice_servers,
                cors_origins=config.effective_allowed_origins,
            )
        )
        webrtc_config = WebRTCGatewayConfig(
            control_endpoint=config.control_endpoint,
            telemetry_endpoint=config.telemetry_endpoint,
            shm_name=config.shm_name,
            shm_slot_size=config.shm_slot_size,
            shm_capacity=config.shm_capacity,
            enable_depth_track=config.enable_depth_track,
            rgb_fps=config.rgb_fps,
            depth_fps=config.depth_fps,
            telemetry_hz=config.telemetry_hz,
            ice_servers=config.ice_servers,
            cors_origins=config.effective_allowed_origins,
        )
        self.session_manager = session_manager or PeerSessionManager(webrtc_config, self.subscriber)
        self.log_tailer = log_tailer or LogTailer(config.process_log_dir)
        self.state_aggregator = state_aggregator or StateAggregator(
            config,
            process_manager=self.process_manager,
            subscriber=self.subscriber,
            control_client=self.control_client,
            session_manager=self.session_manager,
            log_tailer=self.log_tailer,
        )

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self._cors_middleware])
        app["dashboard_web_app"] = self
        app.router.add_get("/api/bootstrap", self.handle_bootstrap)
        app.router.add_get("/api/state", self.handle_state)
        app.router.add_get("/api/logs", self.handle_logs)
        app.router.add_get("/api/events", self.handle_events)
        app.router.add_post("/api/session/start", self.handle_session_start)
        app.router.add_post("/api/session/stop", self.handle_session_stop)
        app.router.add_post("/api/runtime/task", self.handle_runtime_task)
        app.router.add_post("/api/runtime/cancel", self.handle_runtime_cancel)
        app.add_subapp(
            "/api/webrtc",
            WebRTCGateway(
                WebRTCGatewayConfig(
                    control_endpoint=self.config.control_endpoint,
                    telemetry_endpoint=self.config.telemetry_endpoint,
                    shm_name=self.config.shm_name,
                    shm_slot_size=self.config.shm_slot_size,
                    shm_capacity=self.config.shm_capacity,
                    enable_depth_track=self.config.enable_depth_track,
                    rgb_fps=self.config.rgb_fps,
                    depth_fps=self.config.depth_fps,
                    telemetry_hz=self.config.telemetry_hz,
                    ice_servers=self.config.ice_servers,
                    cors_origins=self.config.effective_allowed_origins,
                ),
                subscriber=self.subscriber,
                session_manager=self.session_manager,
            ).create_app(),
        )
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        if self.config.dist_dir.exists():
            app.router.add_get("/", self.handle_index)
            app.router.add_get("/{tail:.*}", self.handle_static)
        return app

    @property
    def _cors_middleware(self):
        @web.middleware
        async def middleware(request, handler):  # noqa: ANN001
            try:
                if request.method == "OPTIONS":
                    response = web.Response(status=200)
                else:
                    response = await handler(request)
            except web.HTTPException as exc:
                response = web.Response(status=exc.status, text=exc.reason, headers=exc.headers)
            self._apply_cors_headers(request, response)
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

        return middleware

    def _apply_cors_headers(self, request, response) -> None:  # noqa: ANN001
        origin = request.headers.get("Origin", "")
        origin_header = cors_header_for_origin(origin, self.config.effective_allowed_origins)
        if origin_header is None:
            return
        response.headers["Access-Control-Allow-Origin"] = origin_header
        if origin_header != "*":
            response.headers["Vary"] = "Origin"

    async def _on_startup(self, app) -> None:  # noqa: ANN001
        _ = app
        await self.subscriber.start()
        await self.state_aggregator.start()

    async def _on_cleanup(self, app) -> None:  # noqa: ANN001
        _ = app
        await self.state_aggregator.close()
        await self.process_manager.stop_all()
        self.control_client.close()

    async def handle_bootstrap(self, request) -> web.Response:  # noqa: ANN001
        _ = request
        return web.json_response(
            {
                "plannerModes": ["interactive", "pointgoal"],
                "launchModes": ["gui", "headless"],
                "scenePresets": ["warehouse", "interioragent", "interior agent kujiale 3"],
                "apiBaseUrl": self.config.api_base_url,
                "devOrigin": self.config.dev_origin,
                "webrtcBasePath": self.config.webrtc_base_url,
            }
        )

    async def handle_state(self, request) -> web.Response:  # noqa: ANN001
        _ = request
        await self.state_aggregator.force_refresh()
        return web.json_response(self.state_aggregator.snapshot())

    async def handle_logs(self, request) -> web.Response:  # noqa: ANN001
        limit = int(request.query.get("limit", "200"))
        return web.json_response({"logs": self.state_aggregator.get_recent_logs(limit=max(limit, 1))})

    async def handle_events(self, request) -> web.StreamResponse:  # noqa: ANN001
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        self._apply_cors_headers(request, response)
        await response.prepare(request)
        queue = self.state_aggregator.add_listener()
        try:
            if not await self._write_sse_chunk(response, self._sse_message("state", self.state_aggregator.snapshot()).encode("utf-8")):
                return response
            while True:
                transport = request.transport
                if transport is None or transport.is_closing():
                    break
                try:
                    state = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    if not await self._write_sse_chunk(response, b": keepalive\n\n"):
                        break
                    continue
                if not await self._write_sse_chunk(response, self._sse_message("state", state).encode("utf-8")):
                    break
        finally:
            self.state_aggregator.remove_listener(queue)
            with suppress(Exception):
                await response.write_eof()
        return response

    async def handle_session_start(self, request) -> web.Response:  # noqa: ANN001
        payload = await request.json()
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="session start payload must be a JSON object")
        try:
            session_request = parse_session_request(payload)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        try:
            await self.process_manager.start_session(session_request)
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        await self.state_aggregator.force_refresh()
        return web.json_response(self.state_aggregator.snapshot())

    async def handle_session_stop(self, request) -> web.Response:  # noqa: ANN001
        _ = request
        await self.process_manager.stop_all()
        await self.state_aggregator.force_refresh()
        return web.json_response(self.state_aggregator.snapshot())

    async def handle_runtime_task(self, request) -> web.Response:  # noqa: ANN001
        payload = await request.json()
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="task payload must be a JSON object")
        current = self.process_manager.current_request
        if current is None or current.planner_mode != "interactive":
            raise web.HTTPConflict(reason="interactive runtime session is not active")
        instruction = str(payload.get("instruction", "")).strip()
        if instruction == "":
            raise web.HTTPBadRequest(reason="instruction is required")
        task = self.control_client.submit_task(instruction)
        await self.state_aggregator.force_refresh()
        return web.json_response({"taskId": task.task_id, "commandText": task.command_text})

    async def handle_runtime_cancel(self, request) -> web.Response:  # noqa: ANN001
        _ = request
        current = self.process_manager.current_request
        if current is None or current.planner_mode != "interactive":
            raise web.HTTPConflict(reason="interactive runtime session is not active")
        control = self.control_client.cancel_interactive_task()
        await self.state_aggregator.force_refresh()
        return web.json_response({"requestId": control.request_id, "action": control.action})

    async def handle_index(self, request) -> web.FileResponse:  # noqa: ANN001
        _ = request
        return web.FileResponse(self.config.dist_dir / "index.html")

    async def handle_static(self, request) -> web.StreamResponse:  # noqa: ANN001
        tail = str(request.match_info.get("tail", ""))
        if tail.startswith("api/"):
            raise web.HTTPNotFound()
        candidate = (self.config.dist_dir / tail).resolve()
        if str(candidate).startswith(str(self.config.dist_dir.resolve())) and candidate.is_file():
            return web.FileResponse(candidate)
        return web.FileResponse(self.config.dist_dir / "index.html")

    @staticmethod
    def _sse_message(event: str, payload: dict[str, object]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}\n\n"

    @staticmethod
    async def _write_sse_chunk(response, payload: bytes) -> bool:  # noqa: ANN001
        try:
            await response.write(payload)
        except Exception:  # noqa: BLE001
            return False
        return True
