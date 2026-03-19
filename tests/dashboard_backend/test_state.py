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
