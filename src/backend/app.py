"""aiohttp backend application."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from aiohttp import ClientSession, web

from backend.app_keys import (
    API_BASE_URL,
    BROADCAST_TASK,
    CONTROL_RUNTIME_URL,
    DEV_ORIGIN,
    HTTP,
    INFERENCE_SYSTEM_URL,
    NAVIGATION_SYSTEM_URL,
    PLANNER_SYSTEM_URL,
    ROOT_DIR,
    RUNTIME_OWNED,
    RUNTIME_SERVICE,
    RUNTIME_URL,
    RUNTIME_SUBMIT_TASKS,
    SESSION_MANAGER,
    SSE,
    WEBRTC_SERVICE,
    WEBRTC_PROXY_BASE,
)
from backend.occupancy import build_occupancy_payload, handle_image
from backend.session_manager import DashboardSessionManager
from backend.sse import SseBroadcaster
from backend.webrtc import WebRTCService, WebRTCServiceConfig
from backend.webrtc_proxy import get_config, proxy_offer
from backend.models import build_bootstrap_data
from runtime.service import RuntimeService


async def _json_body(request: web.Request) -> dict[str, object]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def create_app(
    *,
    root_dir: str,
    api_base_url: str,
    dev_origin: str,
    inference_system_url: str,
    planner_system_url: str,
    navigation_system_url: str,
    control_runtime_url: str,
    runtime_url: str = "",
    webrtc_proxy_base: str = "",
    runtime_service: RuntimeService | None = None,
    webrtc_service: WebRTCService | None = None,
    webrtc_config: WebRTCServiceConfig | None = None,
) -> web.Application:
    app = web.Application()
    resolved_runtime_url = str(runtime_url).rstrip("/")
    runtime_owned = resolved_runtime_url == ""
    app[ROOT_DIR] = str(root_dir)
    app[API_BASE_URL] = str(api_base_url).rstrip("/")
    app[DEV_ORIGIN] = str(dev_origin)
    app[RUNTIME_URL] = resolved_runtime_url
    app[RUNTIME_OWNED] = runtime_owned
    app[RUNTIME_SERVICE] = (
        runtime_service
        if runtime_service is not None
        else (RuntimeService(Path(root_dir)) if runtime_owned else None)
    )
    app[INFERENCE_SYSTEM_URL] = str(inference_system_url).rstrip("/")
    app[PLANNER_SYSTEM_URL] = str(planner_system_url).rstrip("/")
    app[NAVIGATION_SYSTEM_URL] = str(navigation_system_url).rstrip("/")
    app[CONTROL_RUNTIME_URL] = str(control_runtime_url).rstrip("/")
    app[WEBRTC_PROXY_BASE] = str(webrtc_proxy_base).rstrip("/")
    app[WEBRTC_SERVICE] = (
        webrtc_service
        if webrtc_service is not None
        else (None if app[WEBRTC_PROXY_BASE] else WebRTCService(config=webrtc_config))
    )
    app[SSE] = SseBroadcaster()
    app[HTTP] = None
    app[RUNTIME_SUBMIT_TASKS] = set()
    session_manager = DashboardSessionManager(app)
    app[SESSION_MANAGER] = session_manager

    async def broadcast_loop():
        while True:
            state = await session_manager.build_state()
            await app[SSE].publish_state(state)
            await asyncio.sleep(1.0)

    async def on_startup(_app: web.Application):
        app[HTTP] = ClientSession()
        if app[WEBRTC_SERVICE] is not None:
            await app[WEBRTC_SERVICE].start()
        app[BROADCAST_TASK] = asyncio.create_task(broadcast_loop())

    async def on_cleanup(_app: web.Application):
        if app[BROADCAST_TASK] is not None:
            app[BROADCAST_TASK].cancel()
            try:
                await app[BROADCAST_TASK]
            except BaseException:
                pass
        if app[RUNTIME_SUBMIT_TASKS]:
            pending_submits = list(app[RUNTIME_SUBMIT_TASKS])
            for task in pending_submits:
                task.cancel()
            await asyncio.gather(*pending_submits, return_exceptions=True)
        if app[WEBRTC_SERVICE] is not None:
            await app[WEBRTC_SERVICE].close()
        if app[RUNTIME_OWNED] and app[RUNTIME_SERVICE] is not None:
            await asyncio.to_thread(app[RUNTIME_SERVICE].stop_session)
        if app[HTTP] is not None:
            await app[HTTP].close()

    async def submit_planner_task(payload: dict[str, object]) -> None:
        try:
            async with app[HTTP].post(f"{app[PLANNER_SYSTEM_URL]}/planner/task", json=payload) as response:
                planner_payload = await response.json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            session_manager.record_event(f"planner task submission failed: {type(exc).__name__}: {exc}", level="error")
            return

        if response.status >= 400 or not isinstance(planner_payload, dict) or not bool(planner_payload.get("ok", True)):
            error = planner_payload if isinstance(planner_payload, dict) else {"error": "planner returned non-object payload"}
            session_manager.record_event(f"planner task rejected: {error}", level="error")

    async def bootstrap(_request: web.Request) -> web.Response:
        payload = build_bootstrap_data(
            api_base_url=app[API_BASE_URL],
            dev_origin=app[DEV_ORIGIN],
            webrtc_base_path=f"{app[API_BASE_URL]}/api/webrtc",
        )
        return web.json_response(payload)

    async def state(_request: web.Request) -> web.Response:
        return web.json_response(await session_manager.build_state())

    async def events(request: web.Request) -> web.StreamResponse:
        return await app[SSE].subscribe(request)

    async def logs(request: web.Request) -> web.Response:
        state_payload = await session_manager.build_state()
        try:
            limit = max(1, int(request.query.get("limit", "80")))
        except ValueError:
            limit = 80
        logs_payload = list(state_payload.get("logs", []))
        return web.json_response({"logs": logs_payload[-limit:]})

    async def session_start(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        try:
            state_payload = await session_manager.start_session(payload)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        status = 200 if bool(state_payload.get("session", {}).get("active")) else 503
        return web.json_response(state_payload, status=status)

    async def session_stop(_request: web.Request) -> web.Response:
        return web.json_response(await session_manager.stop_session())

    async def runtime_task(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        if app[HTTP] is None:
            raise web.HTTPServiceUnavailable(reason="backend http client unavailable")
        instruction = str(payload.get("instruction", "")).strip()
        if instruction == "":
            raise web.HTTPBadRequest(reason="instruction is required")
        language = str(payload.get("language", "auto")).strip() or "auto"
        task_id = f"planner-{time.time_ns()}"
        planner_payload = {
            "instruction": instruction,
            "language": language,
            "task_id": task_id,
        }
        session_manager.record_event(f"submitted task: {instruction}")
        submit_task = asyncio.create_task(submit_planner_task(planner_payload), name=f"planner-submit-{task_id}")
        app[RUNTIME_SUBMIT_TASKS].add(submit_task)
        submit_task.add_done_callback(app[RUNTIME_SUBMIT_TASKS].discard)
        return web.json_response(
            {
                "ok": True,
                "accepted": True,
                "task_status": "accepted",
                "task_id": task_id,
                "taskId": task_id,
                "instruction": instruction,
                "language": language,
                "commandText": instruction,
            }
        )

    async def runtime_cancel(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        async with app[HTTP].post(f"{app[PLANNER_SYSTEM_URL]}/planner/cancel", json=payload) as response:
            return web.json_response(await response.json(), status=response.status)

    async def occupancy_current(request: web.Request) -> web.Response:
        scene_preset = request.query.get("scenePreset", "warehouse")
        return web.json_response(build_occupancy_payload(scene_preset))

    async def webrtc_config(_request: web.Request) -> web.Response:
        return web.json_response(await get_config(app))

    async def webrtc_offer(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        return await proxy_offer(app, payload)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/events", events)
    app.router.add_get("/api/logs", logs)
    app.router.add_post("/api/session/start", session_start)
    app.router.add_post("/api/session/stop", session_stop)
    app.router.add_post("/api/runtime/task", runtime_task)
    app.router.add_post("/api/runtime/cancel", runtime_cancel)
    app.router.add_get("/api/occupancy/current", occupancy_current)
    app.router.add_get("/api/occupancy/image", handle_image)
    app.router.add_get("/api/webrtc/config", webrtc_config)
    app.router.add_post("/api/webrtc/offer", webrtc_offer)
    return app
