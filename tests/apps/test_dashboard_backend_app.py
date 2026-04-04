from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apps.dashboard_backend_app import parse_args
from dashboard_backend.app import DashboardWebApp
from dashboard_backend.config import DashboardBackendConfig
from dashboard_backend.models import DashboardSessionRequest
from ipc.messages import RuntimeControlRequest, TaskRequest


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeProcessManager:
    def __init__(self) -> None:
        self.current_request: DashboardSessionRequest | None = None
        self.session_started_at: float | None = None
        self.stop_count = 0

    async def start_session(self, request: DashboardSessionRequest) -> None:
        self.current_request = request
        self.session_started_at = 123.0

    async def stop_all(self) -> None:
        self.stop_count += 1
        self.current_request = None
        self.session_started_at = None

    def snapshot(self) -> list[dict[str, object]]:
        request = self.current_request
        names = ("navdp", "system2", "runtime")
        if request is None:
            return [
                {
                    "name": name,
                    "state": "stopped",
                    "required": False,
                    "pid": None,
                    "exitCode": None,
                    "startedAt": None,
                    "healthUrl": "",
                    "stdoutLog": "",
                    "stderrLog": "",
                }
                for name in names
            ]
        return [
            {
                "name": name,
                "state": "running",
                "required": True,
                "pid": 1000 + index,
                "exitCode": None,
                "startedAt": self.session_started_at,
                "healthUrl": "",
                "stdoutLog": f"tmp/{name}.stdout.log",
                "stderrLog": f"tmp/{name}.stderr.log",
            }
            for index, name in enumerate(names)
        ]

    @staticmethod
    def service_urls(name: str) -> tuple[str, str]:
        if name == "navdp":
            return "http://127.0.0.1:8888/health", "http://127.0.0.1:8888/debug_last_input"
        if name == "system2":
            return "http://127.0.0.1:15801/healthz", ""
        return "", ""


class _FakeControlClient:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.idle_count = 0

    def submit_task(self, instruction: str) -> TaskRequest:
        self.submitted.append(instruction)
        return TaskRequest(command_text=instruction)

    def set_idle(self) -> RuntimeControlRequest:
        self.idle_count += 1
        return RuntimeControlRequest(action="set_idle")

    def transport_health_snapshot(self) -> dict[str, object]:
        return {"control_endpoint": "tcp://127.0.0.1:5580", "telemetry_endpoint": "tcp://127.0.0.1:5581"}

    def close(self) -> None:
        return None


class _FakeSubscriber:
    def __init__(self) -> None:
        self.started = False
        self.current_frame = None

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    def last_frame_age_ms(self):
        return None


class _FakeSessionManager:
    def __init__(self) -> None:
        self.active_session = SimpleNamespace(session_id="session-123", track_roles=["rgb"])

    async def accept_offer(self, payload: dict[str, object]):
        return self.active_session, SimpleNamespace(sdp=f"answer:{payload['type']}", type="answer")

    async def close(self) -> None:
        return None


class _FakeLogTailer:
    def get_recent(self, *, limit: int = 200):  # noqa: ANN001
        return [{"source": "runtime", "stream": "stdout", "message": "bridge ready"}][-max(limit, 1) :]


class _FakeStateAggregator:
    def __init__(self, process_manager: _FakeProcessManager) -> None:
        self.process_manager = process_manager
        self.listeners = []
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    async def force_refresh(self) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        request = self.process_manager.current_request
        processes = self.process_manager.snapshot()
        return {
            "timestamp": 100.0,
            "session": {
                "active": request is not None,
                "startedAt": self.process_manager.session_started_at,
                "config": None if request is None else request.to_public_dict(),
                "lastEvent": None,
            },
            "processes": processes,
            "runtime": {},
            "sensors": {},
            "perception": {},
            "memory": {},
            "architecture": {
                "gateway": {"name": "Robot Gateway", "status": "ok", "summary": "frames live", "detail": "", "required": True, "metrics": {}},
                "mainControlServer": {
                    "name": "Main Control Server",
                    "status": "ok",
                    "summary": "task idle" if request is None else "task active",
                    "detail": "",
                    "required": True,
                    "metrics": {"mode": "idle" if request is None else "session_active"},
                    "core": {
                        "worldStateStore": {"name": "World State Store", "status": "ok", "summary": "ready", "detail": "", "required": True, "metrics": {}},
                        "decisionEngine": {"name": "Decision Engine", "status": "ok", "summary": "ready", "detail": "", "required": True, "metrics": {}},
                        "plannerCoordinator": {"name": "Planner Coordinator", "status": "ok", "summary": "ready", "detail": "", "required": True, "metrics": {}},
                        "commandResolver": {"name": "Command Resolver", "status": "ok", "summary": "ready", "detail": "", "required": True, "metrics": {}},
                        "safetySupervisor": {"name": "Safety Supervisor", "status": "ok", "summary": "ready", "detail": "", "required": True, "metrics": {}},
                    },
                },
                "modules": {
                    "perception": {"name": "Perception", "status": "ok", "summary": "0 detections", "detail": "", "required": True, "metrics": {}},
                    "memory": {"name": "Memory", "status": "inactive", "summary": "No active memory task", "detail": "", "required": False, "metrics": {}},
                    "s2": {"name": "S2", "status": "not_required" if request is None else "ok", "summary": "", "detail": "", "required": request is not None, "metrics": {}},
                    "nav": {"name": "Nav", "status": "ok" if request is not None else "inactive", "summary": "", "detail": "", "required": request is not None, "metrics": {}},
                    "locomotion": {"name": "Locomotion", "status": "ok" if request is not None else "inactive", "summary": "", "detail": "", "required": request is not None, "metrics": {}},
                    "telemetry": {"name": "Telemetry", "status": "ok", "summary": "", "detail": "", "required": request is not None, "metrics": {}},
                },
            },
            "services": {
                "navdp": {"name": "navdp", "status": "ok"},
                "system2": {
                    "name": "system2",
                    "status": "inactive" if request is None else "ok",
                    "output": None if request is None else {"decisionMode": "pixel_goal", "rawText": "120, 80"},
                },
            },
            "transport": {"viewerEnabled": False if request is None else request.viewer_enabled},
            "logs": [{"source": "runtime", "stream": "event", "message": "ready"}],
            "selectedTargetSummary": None,
            "latencyBreakdown": {
                "frameAgeMs": None,
                "perceptionLatencyMs": None,
                "memoryLatencyMs": None,
                "s2LatencyMs": None,
                "navLatencyMs": None,
                "locomotionLatencyMs": None,
            },
            "cognitionTrace": [],
            "recoveryTransitions": [],
        }

    def add_listener(self):
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.listeners.append(queue)
        return queue

    def remove_listener(self, queue) -> None:  # noqa: ANN001
        self.listeners = [item for item in self.listeners if item is not queue]

    def get_recent_logs(self, *, limit: int = 200):  # noqa: ANN001
        return self.snapshot()["logs"]


def test_parse_args_exposes_dashboard_defaults() -> None:
    args = parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 18095
    assert args.allow_origin == []
    assert args.control_endpoint == "tcp://127.0.0.1:5580"
    assert args.telemetry_endpoint == "tcp://127.0.0.1:5581"
    assert args.enable_depth_track is True


def test_dashboard_backend_routes_cover_session_runtime_sse_and_webrtc() -> None:
    pytest.importorskip("aiohttp")

    async def scenario() -> None:
        from aiohttp import ClientSession, web

        process_manager = _FakeProcessManager()
        control_client = _FakeControlClient()
        subscriber = _FakeSubscriber()
        session_manager = _FakeSessionManager()
        state_aggregator = _FakeStateAggregator(process_manager)
        port = _free_tcp_port()
        app = DashboardWebApp(
            DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard", port=port),
            process_manager=process_manager,
            control_client=control_client,
            subscriber=subscriber,
            session_manager=session_manager,
            log_tailer=_FakeLogTailer(),
            state_aggregator=state_aggregator,
        ).create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        try:
            async with ClientSession() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/api/bootstrap",
                    headers={"Origin": "http://tauri.localhost"},
                )
                assert response.status == 200
                assert response.headers["Access-Control-Allow-Origin"] == "http://tauri.localhost"
                bootstrap = await response.json()
                assert bootstrap["apiBaseUrl"] == f"http://127.0.0.1:{port}"
                assert bootstrap["webrtcBasePath"] == f"http://127.0.0.1:{port}/api/webrtc"

                events_response = await client.get(
                    f"http://127.0.0.1:{port}/api/events",
                    headers={"Origin": "http://tauri.localhost"},
                )
                first_event = await events_response.content.readuntil(b"\n\n")
                assert b"event: state" in first_event
                payload = json.loads(first_event.decode("utf-8").split("data:", 1)[1].strip())
                assert payload["session"]["active"] is False
                assert payload["architecture"]["modules"]["s2"]["name"] == "S2"
                assert payload["services"]["system2"]["status"] == "inactive"
                events_response.close()

                response = await client.post(
                    f"http://127.0.0.1:{port}/api/session/start",
                    headers={"Origin": "tauri://localhost"},
                    json={
                        "launchMode": "gui",
                        "scenePreset": "warehouse",
                        "viewerEnabled": True,
                        "memoryStore": True,
                        "detectionEnabled": True,
                        "locomotionConfig": {
                            "actionScale": 0.65,
                            "onnxDevice": "cuda",
                            "cmdMaxVx": 0.8,
                            "cmdMaxVy": 0.4,
                            "cmdMaxWz": 1.0,
                        },
                    },
                )
                assert response.status == 200
                started = await response.json()
                assert started["session"]["active"] is True
                assert started["architecture"]["modules"]["s2"]["status"] == "ok"
                assert started["services"]["system2"]["status"] == "ok"

                response = await client.post(
                    f"http://127.0.0.1:{port}/api/runtime/task",
                    headers={"Origin": "tauri://localhost"},
                    json={"instruction": "go to the loading dock"},
                )
                assert response.status == 200
                task_body = await response.json()
                assert task_body["commandText"] == "go to the loading dock"
                assert control_client.submitted == ["go to the loading dock"]

                response = await client.post(
                    f"http://127.0.0.1:{port}/api/runtime/cancel",
                    headers={"Origin": "tauri://localhost"},
                    json={},
                )
                assert response.status == 200
                cancel_body = await response.json()
                assert cancel_body["action"] == "set_idle"
                assert control_client.idle_count == 1

                response = await client.post(
                    f"http://127.0.0.1:{port}/api/webrtc/offer",
                    headers={"Origin": "tauri://localhost"},
                    json={"sdp": "offer-sdp", "type": "offer"},
                )
                assert response.status == 200
                offer_body = await response.json()
                assert offer_body["sessionId"] == "session-123"

                response = await client.post(
                    f"http://127.0.0.1:{port}/api/session/stop",
                    headers={"Origin": "tauri://localhost"},
                    json={},
                )
                assert response.status == 200
                stopped = await response.json()
                assert stopped["session"]["active"] is False
                assert process_manager.stop_count == 1
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_dashboard_backend_returns_service_unavailable_when_process_start_fails() -> None:
    pytest.importorskip("aiohttp")

    class _FailingProcessManager(_FakeProcessManager):
        async def start_session(self, request: DashboardSessionRequest) -> None:
            _ = request
            raise RuntimeError("navdp failed to start: bind error")

    async def scenario() -> None:
        from aiohttp import ClientSession, web

        app = DashboardWebApp(
            DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
            process_manager=_FailingProcessManager(),
            control_client=_FakeControlClient(),
            subscriber=_FakeSubscriber(),
            session_manager=_FakeSessionManager(),
            log_tailer=_FakeLogTailer(),
            state_aggregator=_FakeStateAggregator(_FakeProcessManager()),
        ).create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_tcp_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()

        try:
            async with ClientSession() as client:
                response = await client.post(
                    f"http://127.0.0.1:{port}/api/session/start",
                    headers={"Origin": "tauri://localhost"},
                    json={
                        "launchMode": "gui",
                        "scenePreset": "warehouse",
                        "viewerEnabled": True,
                        "memoryStore": True,
                        "detectionEnabled": True,
                        "locomotionConfig": {
                            "actionScale": 0.65,
                            "onnxDevice": "cuda",
                            "cmdMaxVx": 0.8,
                            "cmdMaxVy": 0.4,
                            "cmdMaxWz": 1.0,
                        },
                    },
                )
                assert response.status == 503
                assert response.headers["Access-Control-Allow-Origin"] == "tauri://localhost"
                assert "navdp failed to start" in await response.text()
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
