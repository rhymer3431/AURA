from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_backend.config import DashboardBackendConfig
from dashboard_backend.state import StateAggregator
from schemas.recovery import RecoveryStateSnapshot
from schemas.world_state import (
    PlanningStateSnapshot,
    RuntimeStateSnapshot,
    SafetyStateSnapshot,
    TaskSnapshot,
    WorldStateSnapshot,
)


class _FakeSubscriber:
    def __init__(self) -> None:
        self.listener = asyncio.Queue()

    def add_listener(self):
        return self.listener

    def remove_listener(self, queue) -> None:  # noqa: ANN001
        _ = queue

    @property
    def current_frame(self):
        return None

    def last_frame_age_ms(self):
        return None


class _FakeProcessManager:
    current_request = None
    session_started_at = None

    def snapshot(self):
        return []


class _FakeControlClient:
    @staticmethod
    def transport_health_snapshot() -> dict[str, object]:
        return {}


class _FakeSessionManager:
    active_session = None


class _FakeLogTailer:
    @staticmethod
    def get_recent(*, limit: int = 200):  # noqa: ARG004
        return []


class _FakeSessionRequest:
    viewer_enabled = True

    @staticmethod
    def required_process_names() -> set[str]:
        return {"navdp", "system2", "dual", "runtime"}

    @staticmethod
    def to_public_dict() -> dict[str, object]:
        return {"viewerEnabled": True}


def test_state_aggregator_start_cleans_up_client_session_on_refresh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    aggregator = StateAggregator(
        DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
        process_manager=_FakeProcessManager(),
        subscriber=_FakeSubscriber(),
        control_client=_FakeControlClient(),
        session_manager=_FakeSessionManager(),
        log_tailer=_FakeLogTailer(),
    )

    async def boom() -> None:
        raise RuntimeError("startup refresh failed")

    monkeypatch.setattr(aggregator, "force_refresh", boom)

    with pytest.raises(RuntimeError, match="startup refresh failed"):
        asyncio.run(aggregator.start())

    assert aggregator._client is None
    assert aggregator._tasks == []


def test_state_aggregator_builds_runtime_state_from_world_snapshot() -> None:
    aggregator = StateAggregator(
        DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
        process_manager=_FakeProcessManager(),
        subscriber=_FakeSubscriber(),
        control_client=_FakeControlClient(),
        session_manager=_FakeSessionManager(),
        log_tailer=_FakeLogTailer(),
    )

    aggregator._consume_gateway_event(
        "health",
        {
            "component": "aura_runtime",
            "details": {
                "worldState": WorldStateSnapshot(
                    task=TaskSnapshot(task_id="task-1", instruction="dock", mode="NAV", state="active", command_id=7),
                    mode="NAV",
                    planning=PlanningStateSnapshot(
                        plan_version=4,
                        planner_mode="NAV",
                        active_instruction="dock",
                        route_state={"pixelGoal": [10, 20]},
                    ),
                    safety=SafetyStateSnapshot(
                        stale=True,
                        recovery_state=RecoveryStateSnapshot(
                            current_state="REPLAN_PENDING",
                            entered_at_ns=10,
                            retry_count=1,
                            backoff_until_ns=20,
                            last_trigger_reason="trajectory_stale",
                        ),
                    ),
                    runtime=RuntimeStateSnapshot(viewer_publish=True),
                ).to_dict()
            },
        },
    )
    state = aggregator._build_state()

    assert "_runtime_snapshot" not in aggregator.__dict__
    assert state["runtime"]["modes"]["executionMode"] == "NAV"
    assert state["runtime"]["activeInstruction"] == "dock"
    assert state["runtime"]["recoveryState"] == "REPLAN_PENDING"
    assert state["transport"]["viewerPublish"] is True
    assert state["architecture"]["gateway"]["name"] == "Robot Gateway"
    assert state["architecture"]["mainControlServer"]["metrics"]["recoveryState"] == "REPLAN_PENDING"
    assert state["architecture"]["modules"]["s2"]["name"] == "S2"
    assert "dual" not in state["architecture"]["modules"]


def test_state_aggregator_normalizes_system2_output_from_dual_debug() -> None:
    process_manager = _FakeProcessManager()
    process_manager.current_request = _FakeSessionRequest()
    process_manager.session_started_at = 100.0
    process_manager.snapshot = lambda: [  # type: ignore[method-assign]
        {
            "name": "system2",
            "state": "running",
            "required": True,
            "pid": 4242,
            "exitCode": None,
            "startedAt": 100.0,
            "healthUrl": "http://127.0.0.1:8080/health",
            "stdoutLog": "tmp/system2.stdout.log",
            "stderrLog": "tmp/system2.stderr.log",
        }
    ]

    aggregator = StateAggregator(
        DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
        process_manager=process_manager,
        subscriber=_FakeSubscriber(),
        control_client=_FakeControlClient(),
        session_manager=_FakeSessionManager(),
        log_tailer=_FakeLogTailer(),
    )
    aggregator._service_state["dual"] = {
        "name": "dual",
        "status": "ok",
        "debug": {
            "instruction": "dock at the charging station",
            "stats": {
                "last_s2_reason": "121, 84",
                "last_s2_requested_stop": False,
                "last_s2_effective_stop": False,
                "last_s2_mode": "pixel_goal",
                "last_s2_history_frame_ids": [17, 21],
                "last_s2_needs_requery": False,
                "last_s2_raw_text": "121, 84",
                "last_s2_latency_ms": 38.5,
            },
            "system2_session": {
                "instruction": "dock at the charging station",
                "last_output": "121, 84",
                "last_reason": "121, 84",
                "last_history_frame_ids": [17, 21],
                "last_decision_mode": "pixel_goal",
                "last_needs_requery": False,
            },
        },
    }

    state = aggregator._build_state()
    system2 = state["services"]["system2"]

    assert system2["status"] == "ok"
    assert system2["latencyMs"] == 38.5
    assert system2["output"] == {
        "rawText": "121, 84",
        "reason": "121, 84",
        "decisionMode": "pixel_goal",
        "needsRequery": False,
        "historyFrameIds": [17, 21],
        "requestedStop": False,
        "effectiveStop": False,
        "instruction": "dock at the charging station",
        "latencyMs": 38.5,
    }
    assert state["architecture"]["modules"]["s2"]["summary"] == "Decision pixel goal"


def test_state_aggregator_keeps_system2_output_null_before_first_decision() -> None:
    process_manager = _FakeProcessManager()
    process_manager.current_request = _FakeSessionRequest()
    process_manager.session_started_at = 100.0
    process_manager.snapshot = lambda: [  # type: ignore[method-assign]
        {
            "name": "system2",
            "state": "running",
            "required": True,
            "pid": 4242,
            "exitCode": None,
            "startedAt": 100.0,
            "healthUrl": "http://127.0.0.1:8080/health",
            "stdoutLog": "tmp/system2.stdout.log",
            "stderrLog": "tmp/system2.stderr.log",
        }
    ]

    aggregator = StateAggregator(
        DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
        process_manager=process_manager,
        subscriber=_FakeSubscriber(),
        control_client=_FakeControlClient(),
        session_manager=_FakeSessionManager(),
        log_tailer=_FakeLogTailer(),
    )
    aggregator._service_state["dual"] = {
        "name": "dual",
        "status": "ok",
        "debug": {
            "instruction": "dock at the charging station",
            "stats": {
                "last_s2_requested_stop": False,
                "last_s2_effective_stop": False,
                "last_s2_mode": "wait",
                "last_s2_history_frame_ids": [],
                "last_s2_needs_requery": False,
                "last_s2_raw_text": "",
                "last_s2_latency_ms": 0.0,
            },
            "system2_session": {
                "instruction": "dock at the charging station",
                "last_output": "",
                "last_reason": "",
                "last_history_frame_ids": [],
                "last_decision_mode": "wait",
                "last_needs_requery": False,
            },
        },
    }

    state = aggregator._build_state()
    system2 = state["services"]["system2"]

    assert system2["status"] == "awaiting_first_decision"
    assert system2["output"] is None
