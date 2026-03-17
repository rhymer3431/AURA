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


@pytest.mark.parametrize(
    ("component", "compatibility_alias"),
    [
        ("navigation_runtime", None),
        ("aura_runtime", "aura_runtime"),
    ],
)
def test_state_aggregator_normalizes_runtime_owner_metadata(component: str, compatibility_alias: str | None) -> None:
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
            "component": component,
            "details": {
                "snapshot": {
                    "planner": {"plannerControlMode": "interactive"},
                }
            },
        },
    )
    aggregator._consume_gateway_event(
        "notice",
        {
            "component": component,
            "level": "info",
            "notice": "runtime ready",
        },
    )

    state = aggregator._build_state()

    assert state["runtime"]["ownerComponent"] == "navigation_runtime"
    assert state["runtime"]["ownerDisplayName"] == "NavigationRuntime"
    assert state["runtime"]["ownerModulePath"] == "runtime.navigation_runtime"
    assert state["runtime"]["plannerControlMode"] == "interactive"
    if compatibility_alias is None:
        assert "ownerCompatibilityAlias" not in state["runtime"]
    else:
        assert state["runtime"]["ownerCompatibilityAlias"] == compatibility_alias
    assert aggregator._event_logs[-1]["source"] == "navigation_runtime"
