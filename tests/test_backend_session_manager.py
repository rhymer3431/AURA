from __future__ import annotations

import asyncio

from backend.app_keys import (
    API_BASE_URL,
    CONTROL_RUNTIME_URL,
    HTTP,
    INFERENCE_SYSTEM_URL,
    NAVIGATION_SYSTEM_URL,
    PLANNER_SYSTEM_URL,
    RUNTIME_OWNED,
    RUNTIME_SERVICE,
    RUNTIME_URL,
    WEBRTC_SERVICE,
)
from backend.session_manager import DashboardSessionManager


class _FakeRuntimeService:
    def state_payload(self):
        return {
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


def test_session_manager_caches_probe_results_between_nearby_state_requests(monkeypatch) -> None:
    calls = {
        "runtime": 0,
        "planner": 0,
        "navigation": 0,
    }

    def _runtime_status(_base_url: str):
        calls["runtime"] += 1
        return {"ok": False, "error": "offline"}

    def _planner_status(_base_url: str):
        calls["planner"] += 1
        return {"ok": False, "error": "offline"}

    def _navigation_status(_base_url: str):
        calls["navigation"] += 1
        return {"ok": False, "error": "offline"}

    monkeypatch.setattr("backend.session_manager.fetch_runtime_status", _runtime_status)
    monkeypatch.setattr("backend.session_manager.fetch_planner_status", _planner_status)
    monkeypatch.setattr("backend.session_manager.fetch_navigation_status", _navigation_status)

    app = {
        API_BASE_URL: "http://127.0.0.1:18095",
        CONTROL_RUNTIME_URL: "http://127.0.0.1:8892",
        HTTP: None,
        INFERENCE_SYSTEM_URL: "http://127.0.0.1:15880",
        NAVIGATION_SYSTEM_URL: "http://127.0.0.1:17882",
        PLANNER_SYSTEM_URL: "http://127.0.0.1:17881",
        RUNTIME_OWNED: True,
        RUNTIME_SERVICE: _FakeRuntimeService(),
        RUNTIME_URL: "",
        WEBRTC_SERVICE: None,
    }
    manager = DashboardSessionManager(app)

    async def scenario() -> None:
        await manager.build_state(force_refresh=True)
        await manager.build_state()

    asyncio.run(scenario())

    assert calls == {
        "runtime": 1,
        "planner": 1,
        "navigation": 1,
    }
