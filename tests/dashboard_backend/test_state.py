from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dashboard_backend.config import DashboardBackendConfig
from dashboard_backend.state import StateAggregator
from schemas.recovery import RecoveryStateSnapshot
from schemas.world_state import (
    ExecutionStateSnapshot,
    PlanningStateSnapshot,
    RobotStateSnapshot,
    RuntimeStateSnapshot,
    SafetyStateSnapshot,
    TaskSnapshot,
    WorldStateSnapshot,
)


class _FakeSubscriber:
    def __init__(self) -> None:
        self.listener = asyncio.Queue()
        self._current_frame = None

    def add_listener(self):
        return self.listener

    def remove_listener(self, queue) -> None:  # noqa: ANN001
        _ = queue

    @property
    def current_frame(self):
        return self._current_frame

    def last_frame_age_ms(self):
        return None


class _FakeProcessManager:
    current_request = None
    session_started_at = None

    def snapshot(self):
        return []

    @staticmethod
    def service_urls(name: str) -> tuple[str, str]:
        if name == "navdp":
            return "http://127.0.0.1:8888/health", "http://127.0.0.1:8888/debug_last_input"
        if name == "system2":
            return "http://127.0.0.1:15801/healthz", ""
        return "", ""


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
        return {"navdp", "system2", "runtime"}

    @staticmethod
    def to_public_dict() -> dict[str, object]:
        return {"viewerEnabled": True}


def _aggregator(process_manager: _FakeProcessManager | None = None) -> StateAggregator:
    return StateAggregator(
        DashboardBackendConfig(repo_root=ROOT, dashboard_dir=ROOT / "dashboard"),
        process_manager=_FakeProcessManager() if process_manager is None else process_manager,
        subscriber=_FakeSubscriber(),
        control_client=_FakeControlClient(),
        session_manager=_FakeSessionManager(),
        log_tailer=_FakeLogTailer(),
    )


def test_state_aggregator_start_cleans_up_client_session_on_refresh_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    aggregator = _aggregator()

    async def boom() -> None:
        raise RuntimeError("startup refresh failed")

    monkeypatch.setattr(aggregator, "force_refresh", boom)

    with pytest.raises(RuntimeError, match="startup refresh failed"):
        asyncio.run(aggregator.start())

    assert aggregator._client is None
    assert aggregator._tasks == []


def test_state_aggregator_builds_runtime_state_from_world_snapshot() -> None:
    aggregator = _aggregator()

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
                        goal_version=5,
                        traj_version=6,
                        planner_mode="NAV",
                        active_instruction="dock",
                        active_nav_plan={
                            "trajectory_point_count": 2,
                            "trajectory_world": [[0.0, 0.0, 0.0], [0.6, 0.1, 0.0]],
                        },
                        system2_pixel_goal=[120, 80],
                        route_state={"pixelGoal": [10, 20]},
                    ),
                    execution=ExecutionStateSnapshot(
                        locomotion_proposal_summary={
                            "goal_distance_m": 1.4,
                            "command_vector": [0.2, 0.0, 0.1],
                        },
                        active_command_type="NAV_TO_POSE",
                        active_target={"target_track_id": "track-1", "nav_goal_pixel": [100, 60], "source": "perception"},
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
    aggregator.subscriber._current_frame = SimpleNamespace(
        seq=7,
        viewer_overlay={
            "detections": [
                {
                    "class_name": "apple",
                    "track_id": "track-1",
                    "bbox_xyxy": [1, 2, 10, 12],
                    "confidence": 0.9,
                    "depth_m": 1.5,
                    "world_pose_xyz": [0.1, 0.2, 0.3],
                }
            ]
        },
        frame_header=SimpleNamespace(frame_id=11, timestamp_ns=1234, source="unit_test"),
    )
    state = aggregator._build_state()

    assert state["runtime"]["modes"]["executionMode"] == "NAV"
    assert state["runtime"]["activeInstruction"] == "dock"
    assert state["runtime"]["recoveryState"] == "REPLAN_PENDING"
    assert state["runtime"]["navTrajectoryPointCount"] == 2
    assert state["runtime"]["commandVector"] == [0.2, 0.0, 0.1]
    assert state["transport"]["viewerPublish"] is True
    assert state["architecture"]["gateway"]["name"] == "Robot Gateway"
    assert state["architecture"]["mainControlServer"]["metrics"]["recoveryState"] == "REPLAN_PENDING"
    assert state["architecture"]["modules"]["s2"]["name"] == "S2"
    assert state["selectedTargetSummary"]["trackId"] == "track-1"
    assert state["latencyBreakdown"]["navLatencyMs"] is None
    assert len(state["cognitionTrace"]) == 1
    assert state["recoveryTransitions"] == []


def test_state_aggregator_normalizes_system2_output_from_health_payload() -> None:
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
            "healthUrl": "http://127.0.0.1:15801/healthz",
            "stdoutLog": "tmp/system2.stdout.log",
            "stderrLog": "tmp/system2.stderr.log",
        }
    ]

    aggregator = _aggregator(process_manager)
    aggregator._service_state["system2"] = {
        "name": "system2",
        "status": "ok",
        "latencyMs": 38.5,
        "health": {
            "system2_output": {
                "instruction": "dock at the charging station",
                "rawText": "121, 84",
                "decisionMode": "pixel_goal",
                "historyFrameIds": [17, 21],
                "needsRequery": False,
                "latencyMs": 38.5,
            }
        },
        "debug": {},
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
            "healthUrl": "http://127.0.0.1:15801/healthz",
            "stdoutLog": "tmp/system2.stdout.log",
            "stderrLog": "tmp/system2.stderr.log",
        }
    ]

    aggregator = _aggregator(process_manager)
    aggregator._service_state["system2"] = {
        "name": "system2",
        "status": "ok",
        "health": {"system2_output": None},
        "debug": {},
    }

    state = aggregator._build_state()
    system2 = state["services"]["system2"]

    assert system2["status"] == "awaiting_first_decision"
    assert system2["output"] is None


def test_state_aggregator_updates_trace_in_place_for_same_frame_and_tracks_recovery_transitions() -> None:
    aggregator = _aggregator()

    def consume(current_state: str, reason: str, *, decision_mode: str) -> dict[str, object]:
        aggregator._consume_gateway_event(
            "health",
            {
                "component": "aura_runtime",
                "details": {
                    "worldState": WorldStateSnapshot(
                        task=TaskSnapshot(task_id="task-1", instruction="dock", mode="NAV", state="active", command_id=7),
                        mode="NAV",
                        robot=RobotStateSnapshot(frame_id=44),
                        planning=PlanningStateSnapshot(plan_version=4, goal_version=5, traj_version=6, planner_mode="NAV"),
                        execution=ExecutionStateSnapshot(active_command_type="NAV_TO_POSE"),
                        safety=SafetyStateSnapshot(
                            recovery_state=RecoveryStateSnapshot(
                                current_state=current_state,
                                entered_at_ns=10,
                                retry_count=1,
                                backoff_until_ns=20,
                                last_trigger_reason=reason,
                            ),
                        ),
                    ).to_dict()
                },
            },
        )
        aggregator._service_state["system2"] = {
            "name": "system2",
            "status": "ok",
            "health": {
                "system2_output": {
                    "instruction": "dock",
                    "rawText": "120, 80",
                    "decisionMode": decision_mode,
                    "historyFrameIds": [],
                    "needsRequery": False,
                }
            },
            "debug": {},
        }
        return aggregator._build_state()

    first = consume("NORMAL", "clear", decision_mode="pixel_goal")
    second = consume("SAFE_STOP", "trajectory_blocked", decision_mode="stop")

    assert len(first["cognitionTrace"]) == 1
    assert len(second["cognitionTrace"]) == 1
    assert second["cognitionTrace"][0]["s2DecisionMode"] == "stop"
    assert second["cognitionTrace"][0]["recoveryState"] == "SAFE_STOP"
    assert second["recoveryTransitions"] == [
        {
            "from": "NORMAL",
            "to": "SAFE_STOP",
            "reason": "trajectory_blocked",
            "timestamp": second["recoveryTransitions"][0]["timestamp"],
            "retryCount": 1,
        }
    ]
