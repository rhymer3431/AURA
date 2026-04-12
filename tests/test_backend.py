from __future__ import annotations

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from systems.control.runtime_control_api import RuntimeControlApiServer
from backend.app import create_app


async def _make_client(
    *,
    runtime_url: str = "",
    planner_system_url: str = "http://127.0.0.1:17881",
    control_runtime_url: str = "http://127.0.0.1:8892",
    runtime_service=None,
    webrtc_service=None,
) -> TestClient:
    app = create_app(
        root_dir="C:/Users/mango/project/AURA/system",
        api_base_url="http://127.0.0.1:18095",
        dev_origin="http://127.0.0.1:5173",
        runtime_url=runtime_url,
        inference_system_url="http://127.0.0.1:15880",
        planner_system_url=planner_system_url,
        navigation_system_url="http://127.0.0.1:17882",
        control_runtime_url=control_runtime_url,
        runtime_service=runtime_service,
        webrtc_service=webrtc_service,
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


class FakeWebRTCService:
    def __init__(
        self,
        *,
        frame_state: dict[str, object] | None = None,
        frame_meta: dict[str, object] | None = None,
        health_snapshot: dict[str, object] | None = None,
    ) -> None:
        self.subscriber = type(
            "_Subscriber",
            (),
            {
                "build_state_snapshot": lambda _self: frame_state,
                "build_frame_meta": lambda _self: frame_meta,
            },
        )()
        self._health_snapshot = health_snapshot

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        return {
            "iceServers": [],
            "enableDepthTrack": True,
            "transportMode": "webrtc" if enabled else "disabled",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc" if enabled else "disabled",
            "observeOnly": True,
            "peerModel": "single",
            "channelLabels": ["state", "telemetry"],
            "proxyMode": "internal",
        }

    async def accept_offer(self, offer_payload: dict[str, object]) -> dict[str, object]:
        return {
            "sdp": f"answer-for-{offer_payload.get('type', 'offer')}",
            "type": "answer",
            "sessionId": "peer-test",
        }

    def health_snapshot(self) -> dict[str, object]:
        if self._health_snapshot is not None:
            return dict(self._health_snapshot)
        return {
            "transport": "webrtc",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc",
            "frameAvailable": False,
            "streamStalled": False,
            "frameSeq": None,
            "frameId": None,
            "frameAgeMs": None,
            "lastGoodFrameAgeMs": None,
            "peerActive": False,
            "peerSessionId": None,
            "peerTrackRoles": [],
            "rgbAvailable": False,
            "depthAvailable": False,
            "source": "control_runtime",
            "image": {"width": 0, "height": 0},
            "dropCounters": {"shmOverwrite": 0},
            "transportHealth": {
                "control_endpoint": "tcp://127.0.0.1:5580",
                "telemetry_endpoint": "tcp://127.0.0.1:5581",
                "shm_name": "aura_viewer_shm_01",
                "decodeOk": 0,
                "decodeDrops": 0,
                "shmOverwriteDrops": 0,
                "staleTransitions": 0,
            },
        }


def test_backend_bootstrap_and_degraded_state_contracts() -> None:
    async def scenario():
        client = await _make_client(webrtc_service=FakeWebRTCService())
        try:
            bootstrap = await client.get("/api/bootstrap")
            assert bootstrap.status == 200
            bootstrap_payload = await bootstrap.json()
            assert bootstrap_payload["apiBaseUrl"] == "http://127.0.0.1:18095"
            assert "scenePresets" in bootstrap_payload

            state = await client.get("/api/state")
            assert state.status == 200
            state_payload = await state.json()
            assert state_payload["session"]["config"] is None
            assert state_payload["processes"] == []
            assert state_payload["services"]["backend"]["status"] == "healthy"
            assert state_payload["services"]["runtime"]["status"] == "healthy"
            assert state_payload["services"]["runtime"]["health"]["ownedByBackend"] is True
            assert state_payload["services"]["controlRuntime"]["status"] == "inactive"
            assert state_payload["services"]["inferenceSystem"]["status"] == "inactive"
            assert state_payload["services"]["navigationSystem"]["status"] == "inactive"
            assert state_payload["services"]["plannerSystem"]["status"] == "inactive"

            events = await client.get("/api/events")
            assert events.status == 200
            assert events.headers["Content-Type"].startswith("text/event-stream")

            occupancy = await client.get("/api/occupancy/current?scenePreset=warehouse")
            assert occupancy.status == 200
            occupancy_payload = await occupancy.json()
            assert occupancy_payload["available"] is False

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert "iceServers" in webrtc_payload
            assert webrtc_payload["transportMode"] == "disabled"
            assert webrtc_payload["mediaIngress"] == "zmq+shm"
            assert webrtc_payload["proxyMode"] == "internal"
            assert state_payload["transport"]["busHealth"]["control_endpoint"] == "tcp://127.0.0.1:5580"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_proxies_runtime_session_routes() -> None:
    async def scenario():
        session_state = {
            "ok": True,
            "session": {
                "active": False,
                "state": "inactive",
                "startedAt": None,
                "config": None,
                "lastEvent": None,
            },
            "processes": [],
            "serviceEndpoints": {},
            "lastError": None,
        }

        async def healthz(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True, "service": "runtime"})

        async def state_route(_request: web.Request) -> web.Response:
            return web.json_response(session_state)

        async def start_route(request: web.Request) -> web.Response:
            payload = await request.json()
            session_state["session"] = {
                "active": True,
                "state": "running",
                "startedAt": 123.0,
                "config": payload,
                "lastEvent": {"message": "runtime session running"},
            }
            session_state["processes"] = [
                {
                    "name": "control_runtime",
                    "state": "running",
                    "required": True,
                    "pid": 1234,
                    "exitCode": None,
                    "startedAt": 123.0,
                    "healthUrl": "http://127.0.0.1:8892/healthz",
                    "stdoutLog": "control.stdout.log",
                    "stderrLog": "control.stderr.log",
                }
            ]
            return web.json_response(session_state)

        async def stop_route(_request: web.Request) -> web.Response:
            session_state["session"] = {
                "active": False,
                "state": "inactive",
                "startedAt": None,
                "config": None,
                "lastEvent": {"message": "runtime session stopped"},
            }
            return web.json_response(session_state)

        supervisor_app = web.Application()
        supervisor_app.router.add_get("/healthz", healthz)
        supervisor_app.router.add_get("/session/state", state_route)
        supervisor_app.router.add_post("/session/start", start_route)
        supervisor_app.router.add_post("/session/stop", stop_route)
        supervisor_server = TestServer(supervisor_app)
        supervisor_client = TestClient(supervisor_server)
        await supervisor_client.start_server()

        client = await _make_client(
            runtime_url=str(supervisor_client.make_url("")).rstrip("/"),
            webrtc_service=FakeWebRTCService(),
        )
        try:
            start = await client.post("/api/session/start", json={"launchMode": "headless", "viewerEnabled": True})
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["session"]["active"] is True
            assert start_payload["services"]["runtime"]["status"] == "healthy"
            assert start_payload["services"]["backend"]["status"] == "healthy"
            assert start_payload["services"]["controlRuntime"]["status"] == "degraded"
            assert start_payload["processes"][0]["name"] == "control_runtime"

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert webrtc_payload["transportMode"] == "webrtc"

            stop = await client.post("/api/session/stop", json={})
            assert stop.status == 200
            stop_payload = await stop.json()
            assert stop_payload["session"]["active"] is False
        finally:
            await client.close()
            await supervisor_client.close()

    asyncio.run(scenario())


def test_backend_owns_runtime_session_routes_by_default() -> None:
    class FakeRuntimeService:
        def __init__(self) -> None:
            self.last_start_config: dict[str, object] | None = None
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def state_payload(self, *, ok: bool = True):
            payload = dict(self._state)
            payload["ok"] = ok
            return payload

        def start_session(self, config: dict[str, object]):
            self.last_start_config = dict(config)
            self._state = {
                "ok": True,
                "session": {
                    "active": True,
                    "state": "running",
                    "startedAt": 123.0,
                    "config": dict(config),
                    "lastEvent": {"message": "runtime session running"},
                },
                "processes": [
                    {
                        "name": "control_runtime",
                        "state": "running",
                        "required": True,
                        "pid": 1234,
                        "exitCode": None,
                        "startedAt": 123.0,
                        "healthUrl": "http://127.0.0.1:8892/healthz",
                        "stdoutLog": "control.stdout.log",
                        "stderrLog": "control.stderr.log",
                    }
                ],
                "serviceEndpoints": {"controlRuntimeUrl": "http://127.0.0.1:8892"},
                "lastError": None,
            }
            return self.state_payload()

        def stop_session(self):
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime session stopped"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }
            return self.state_payload()

    async def scenario():
        runtime_service = FakeRuntimeService()
        webrtc_service = FakeWebRTCService(
            frame_meta={
                "type": "frame_meta",
                "frame_id": 3,
                "detections": [],
                "trajectoryPixels": [[12, 34], [56, 78]],
                "activeTarget": {
                    "className": "Navigation Goal",
                    "source": "navigation",
                    "nav_goal_pixel": [220, 140],
                    "world_pose_xyz": [1.0, 2.0, 0.0],
                },
                "system2PixelGoal": [220, 140],
            },
            health_snapshot={
                "transport": "webrtc",
                "mediaIngress": "zmq+shm",
                "mediaEgress": "webrtc",
                "frameAvailable": True,
                "streamStalled": False,
                "frameSeq": 3,
                "frameId": 3,
                "frameAgeMs": 15.0,
                "lastGoodFrameAgeMs": 15.0,
                "peerActive": False,
                "peerSessionId": None,
                "peerTrackRoles": [],
                "rgbAvailable": True,
                "depthAvailable": False,
                "source": "control_runtime",
                "image": {"width": 320, "height": 180},
                "dropCounters": {"shmOverwrite": 0},
                "transportHealth": {
                    "control_endpoint": "tcp://127.0.0.1:5580",
                    "telemetry_endpoint": "tcp://127.0.0.1:5581",
                    "shm_name": "aura_viewer_shm_01",
                    "decodeOk": 5,
                    "decodeDrops": 0,
                    "shmOverwriteDrops": 0,
                    "staleTransitions": 0,
                },
            },
        )
        client = await _make_client(runtime_service=runtime_service, webrtc_service=webrtc_service)
        try:
            start = await client.post(
                "/api/session/start",
                json={
                    "launchMode": "headless",
                    "viewerEnabled": True,
                    "memoryStore": True,
                    "detectionEnabled": False,
                    "locomotionConfig": {
                        "actionScale": 0.6,
                        "onnxDevice": "cpu",
                        "cmdMaxVx": 0.4,
                        "cmdMaxVy": 0.2,
                        "cmdMaxWz": 0.7,
                    },
                },
            )
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["session"]["active"] is True
            assert start_payload["services"]["runtime"]["status"] == "healthy"
            assert start_payload["services"]["runtime"]["health"]["ownedByBackend"] is True
            assert start_payload["services"]["backend"]["status"] == "healthy"
            assert start_payload["services"]["controlRuntime"]["status"] == "degraded"
            assert start_payload["services"]["plannerSystem"]["status"] == "degraded"
            assert start_payload["processes"][0]["name"] == "control_runtime"
            assert runtime_service.last_start_config == {
                "launchMode": "headless",
                "scenePreset": "warehouse",
                "viewerEnabled": True,
                "memoryStore": True,
                "detectionEnabled": False,
                "locomotionConfig": {
                    "actionScale": 0.6,
                    "onnxDevice": "cpu",
                    "cmdMaxVx": 0.4,
                    "cmdMaxVy": 0.2,
                    "cmdMaxWz": 0.7,
                },
            }
            assert start_payload["transport"]["frameAvailable"] is True
            assert start_payload["transport"]["streamStalled"] is False
            assert start_payload["transport"]["frameAgeMs"] == 15.0
            assert start_payload["transport"]["lastGoodFrameAgeMs"] == 15.0
            assert start_payload["selectedTargetSummary"]["className"] == "Navigation Goal"
            assert start_payload["selectedTargetSummary"]["navGoalPixel"] == [220, 140]
            assert start_payload["perception"]["trajectoryPointCount"] == 2

            offer = await client.post("/api/webrtc/offer", json={"sdp": "offer", "type": "offer"})
            assert offer.status == 200
            offer_payload = await offer.json()
            assert offer_payload["type"] == "answer"
            assert offer_payload["sessionId"] == "peer-test"

            stop = await client.post("/api/session/stop", json={})
            assert stop.status == 200
            stop_payload = await stop.json()
            assert stop_payload["session"]["active"] is False
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_webrtc_config_disables_transport_when_viewer_publish_is_disabled() -> None:
    class FakeRuntimeService:
        def __init__(self) -> None:
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def state_payload(self, *, ok: bool = True):
            payload = dict(self._state)
            payload["ok"] = ok
            return payload

        def start_session(self, config: dict[str, object]):
            self._state = {
                "ok": True,
                "session": {
                    "active": True,
                    "state": "running",
                    "startedAt": 123.0,
                    "config": dict(config),
                    "lastEvent": {"message": "runtime session running"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }
            return self.state_payload()

        def stop_session(self):
            return self.state_payload()

    async def scenario():
        client = await _make_client(runtime_service=FakeRuntimeService(), webrtc_service=FakeWebRTCService())
        try:
            start = await client.post("/api/session/start", json={"launchMode": "headless", "viewerEnabled": False})
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["transport"]["viewerEnabled"] is False
            assert start_payload["architecture"]["modules"]["telemetry"]["summary"] == "Viewer publish disabled"

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert webrtc_payload["transportMode"] == "disabled"
            assert webrtc_payload["mediaEgress"] == "disabled"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_runtime_routes_proxy_to_planner_system() -> None:
    async def scenario():
        recorded: list[tuple[str, dict[str, object]]] = []
        planner_task_called = asyncio.Event()

        async def planner_task(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(("task", payload))
            planner_task_called.set()
            await asyncio.sleep(0.2)
            return web.json_response({"ok": True, "task_status": "running"})

        async def planner_cancel(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(("cancel", payload))
            return web.json_response({"ok": True, "task_status": "cancelled"})

        planner_app = web.Application()
        planner_app.router.add_post("/planner/task", planner_task)
        planner_app.router.add_post("/planner/cancel", planner_cancel)
        planner_server = TestServer(planner_app)
        planner_client = TestClient(planner_server)
        await planner_client.start_server()

        client = await _make_client(planner_system_url=str(planner_client.make_url("")).rstrip("/"))
        try:
            started_at = asyncio.get_running_loop().time()
            submit = await client.post("/api/runtime/task", json={"instruction": "go to the tv", "language": "en"})
            submit_elapsed = asyncio.get_running_loop().time() - started_at
            assert submit.status == 200
            submit_payload = await submit.json()
            assert submit_payload["task_status"] == "accepted"
            assert submit_payload["taskId"] == submit_payload["task_id"]
            assert submit_payload["commandText"] == "go to the tv"
            assert submit_elapsed < 0.18
            await asyncio.wait_for(planner_task_called.wait(), timeout=1.0)

            cancel = await client.post("/api/runtime/cancel", json={})
            assert cancel.status == 200
            cancel_payload = await cancel.json()
            assert cancel_payload["task_status"] == "cancelled"
        finally:
            await client.close()
            await planner_client.close()

        assert recorded[0] == (
            "task",
            {
                "instruction": "go to the tv",
                "language": "en",
                "task_id": submit_payload["task_id"],
            },
        )
        assert recorded[1] == ("cancel", {})

    asyncio.run(scenario())


def test_dashboard_task_route_forwards_go_to_purple_box_instruction() -> None:
    async def scenario():
        recorded: list[dict[str, object]] = []
        planner_task_called = asyncio.Event()

        async def planner_task(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(payload)
            planner_task_called.set()
            return web.json_response({"ok": True, "task_status": "running"})

        planner_app = web.Application()
        planner_app.router.add_post("/planner/task", planner_task)
        planner_server = TestServer(planner_app)
        planner_client = TestClient(planner_server)
        await planner_client.start_server()

        client = await _make_client(planner_system_url=str(planner_client.make_url("")).rstrip("/"))
        try:
            submit = await client.post("/api/runtime/task", json={"instruction": "go to purple box", "language": "en"})
            assert submit.status == 200
            submit_payload = await submit.json()
            assert submit_payload["task_status"] == "accepted"
            assert submit_payload["commandText"] == "go to purple box"
            await asyncio.wait_for(planner_task_called.wait(), timeout=1.0)
        finally:
            await client.close()
            await planner_client.close()

        assert recorded == [
            {
                "instruction": "go to purple box",
                "language": "en",
                "task_id": submit_payload["task_id"],
            }
        ]

    asyncio.run(scenario())


def test_backend_rejects_invalid_session_config() -> None:
    async def scenario():
        client = await _make_client(runtime_service=None, webrtc_service=FakeWebRTCService())
        try:
            response = await client.post(
                "/api/session/start",
                json={"viewerEnabled": "maybe", "locomotionConfig": {"actionScale": -1.0}},
            )
            assert response.status == 400
            assert "viewerEnabled must be a boolean" in await response.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_runtime_control_api_exposes_runtime_and_camera_routes() -> None:
    class _Handler:
        def runtime_status(self):
            return {
                "executionMode": "NAV",
                "state_label": "waiting",
                "viewer": {
                    "transport": "webrtc",
                    "frameAvailable": True,
                    "frameSeq": 3,
                    "frameId": 3,
                    "frameAgeMs": 15.0,
                    "peerActive": True,
                    "peerSessionId": "peer-test",
                    "peerTrackRoles": ["rgb"],
                    "rgbAvailable": True,
                    "depthAvailable": True,
                    "source": "control_runtime",
                    "image": {"width": 320, "height": 180},
                },
            }

    class _Camera:
        def pitch_status(self):
            return {"ready": True, "target_pitch_deg": 0.0, "applied_pitch_deg": 0.0}

        def set_pitch_deg(self, value: float):
            return value

        def add_pitch_deg(self, value: float):
            return value

    server = RuntimeControlApiServer("127.0.0.1", 0, _Handler(), _Camera())
    try:
        server.start()
        from urllib.request import Request, urlopen

        with urlopen(f"http://127.0.0.1:{server.port}/runtime/status", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["ok"] is True
            assert payload["executionMode"] == "NAV"
            assert payload["viewer"]["frameAvailable"] is True

        request = Request(
            f"http://127.0.0.1:{server.port}/camera/pitch",
            data=json.dumps({"delta_deg": -10.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["ok"] is True
            assert payload["updated"] == "relative"
    finally:
        server.shutdown()
