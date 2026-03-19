from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionCommand, HealthPing, RuntimeControlRequest, RuntimeNotice, TaskRequest
from runtime.aura_runtime import AuraRuntimeCommandSource
from schemas.recovery import RecoveryStateSnapshot
from schemas.world_state import PlanningStateSnapshot, RuntimeStateSnapshot, SafetyStateSnapshot, TaskSnapshot, WorldStateSnapshot


class _FakePlanningSession:
    def __init__(self) -> None:
        self.events: list[str] = []


class _FakeServer:
    def __init__(self, planning_session: _FakePlanningSession, memory_service: _FakeMemoryService) -> None:
        self._planning_session = planning_session
        self._memory_service = memory_service

    def submit_task_request(self, request: TaskRequest):
        self._planning_session.events.extend(
            [
                f"classify:{request.command_text}",
                "submit_task_request",
            ]
        )
        self._memory_service.set_planner_task(
            instruction=request.command_text,
            planner_mode="nav",
            task_state="pending",
            task_id=str(request.task_id),
            command_id=-1,
        )
        return (
            RuntimeNotice(
                component="main_control_server",
                level="info",
                notice="task classified",
                details={"taskId": str(request.task_id), "executionMode": "NAV", "instruction": request.command_text},
            ),
        )

    def set_idle(self, *, source: str):
        self._planning_session.events.append(f"set_idle:{source}")
        self._memory_service.clear_planner_task(
            task_state="cancelled",
            reason=f"set idle via {source}",
        )
        notice = SimpleNamespace(level="info", notice="execution mode set to idle", details={"source": source, "action": "set_idle"})
        return True, notice

    def snapshot(self) -> WorldStateSnapshot:
        return WorldStateSnapshot(
            task=TaskSnapshot(task_id="task-1", instruction="inspect loading dock", mode="NAV", state="active"),
            mode="NAV",
            planning=PlanningStateSnapshot(
                plan_version=4,
                goal_version=2,
                traj_version=3,
                planner_mode="NAV",
                active_instruction="inspect loading dock",
                route_state={"pixelGoal": [24, 18], "plannerControlMode": "trajectory", "plannerControlReason": "route_refresh"},
                planner_control_mode="trajectory",
                global_route={"enabled": True, "active": True, "waypoint_index": 0, "waypoint_count": 2},
            ),
            safety=SafetyStateSnapshot(
                stale=True,
                recovery_state=RecoveryStateSnapshot(
                    current_state="REPLAN_PENDING",
                    entered_at_ns=12,
                    retry_count=1,
                    backoff_until_ns=48,
                    last_trigger_reason="trajectory_stale",
                ),
            ),
            runtime=RuntimeStateSnapshot(
                launch_mode="headless",
                viewer_publish=False,
                native_viewer="off",
                scene_preset="warehouse",
                show_depth=True,
                memory_store=True,
                detection_enabled=True,
                control_endpoint="tcp://127.0.0.1:5580",
                telemetry_endpoint="tcp://127.0.0.1:5581",
                shm_name="g1_view_frames",
                frame_available=True,
            ),
        )


class _FakeMemoryService:
    def __init__(self) -> None:
        self.spatial_store = SimpleNamespace(objects=[1, 2], places=[1])
        self.semantic_store = SimpleNamespace(list=lambda: ["rule-a", "rule-b"])
        self.keyframes = [1, 2, 3]
        self.scratchpad = SimpleNamespace(
            instruction="inspect loading dock",
            planner_mode="nav",
            task_state="active",
            task_id="task-1",
            command_id=-1,
            goal_summary="dock",
            recent_hint="watch pallet",
            next_priority="keep heading",
        )
        self.set_calls: list[dict[str, object]] = []
        self.clear_calls: list[dict[str, object]] = []

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.set_calls.append(dict(kwargs))

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self.clear_calls.append(dict(kwargs))


class _FakeRuntimeReport:
    ready_for_inference = True
    warnings: list[str] = []
    errors: list[str] = []

    @staticmethod
    def as_dict() -> dict[str, object]:
        return {"ready": True}


def _build_source() -> tuple[AuraRuntimeCommandSource, _FakePlanningSession, _FakeMemoryService, list[dict[str, object]], list[HealthPing]]:
    planning_session = _FakePlanningSession()
    memory_service = _FakeMemoryService()
    notices: list[dict[str, object]] = []
    health_events: list[HealthPing] = []
    source = object.__new__(AuraRuntimeCommandSource)
    source.args = Namespace(
        viewer_control_endpoint="tcp://127.0.0.1:5580",
        viewer_telemetry_endpoint="tcp://127.0.0.1:5581",
        viewer_shm_name="g1_view_frames",
        show_depth=True,
        memory_store=True,
        skip_detection=False,
        scene_preset="warehouse",
    )
    source._launch_mode = "headless"
    source._viewer_publish = False
    source._native_viewer = "off"
    source._runtime_io = object()
    source._planning_session = planning_session
    source._executor = SimpleNamespace()
    source._server = _FakeServer(planning_session, memory_service)
    source._supervisor = SimpleNamespace(
        bridge=SimpleNamespace(publish_health=lambda ping: health_events.append(ping)),
        memory_service=memory_service,
        perception_pipeline=SimpleNamespace(
            detector=SimpleNamespace(
                info=SimpleNamespace(backend_name="stub-detector", selected_reason="unit-test"),
                runtime_report=_FakeRuntimeReport(),
            )
        ),
    )
    source._last_viewer_overlay = {"detections": [{"class_name": "apple"}], "trajectory_pixels": [[1, 2], [3, 4]]}
    source._pending_status = None
    source._active_command = ActionCommand(action_type="LOCAL_SEARCH")
    source._last_runtime_snapshot_frame = -1
    source._publish_notice = lambda *, level, notice, details=None: notices.append(  # type: ignore[method-assign]
        {"level": level, "notice": notice, "details": dict(details or {})}
    )
    return source, planning_session, memory_service, notices, health_events


def test_handle_task_request_routes_dashboard_instruction_into_server_owned_flow() -> None:
    source, planning_session, memory_service, notices, _ = _build_source()

    source._handle_task_request(TaskRequest(command_text="go to the loading dock", task_id="task-1"))

    assert planning_session.events == [
        "classify:go to the loading dock",
        "submit_task_request",
    ]
    assert memory_service.set_calls == [
        {
            "instruction": "go to the loading dock",
            "planner_mode": "nav",
            "task_state": "pending",
            "task_id": "task-1",
            "command_id": -1,
        }
    ]
    assert notices[-1]["notice"] == "task classified"
    assert notices[-1]["details"]["taskId"] == "task-1"


def test_handle_runtime_control_sets_runtime_idle() -> None:
    source, planning_session, memory_service, notices, _ = _build_source()

    source._handle_runtime_control(RuntimeControlRequest(action="set_idle"))

    assert planning_session.events == ["set_idle:dashboard"]
    assert memory_service.clear_calls == [
        {
            "task_state": "cancelled",
            "reason": "set idle via dashboard",
        }
    ]
    assert [item["notice"] for item in notices] == ["execution mode set to idle"]


def test_publish_runtime_snapshot_contains_world_state_and_legacy_contract() -> None:
    source, _, _, _, health_events = _build_source()

    source._publish_runtime_snapshot(frame_idx=5)

    assert len(health_events) == 1
    payload = health_events[0].details
    assert set(payload) == {"worldState", "snapshot"}
    assert payload["worldState"]["task"]["task_id"] == "task-1"
    assert payload["snapshot"]["modes"]["executionMode"] == "NAV"
    assert payload["snapshot"]["planner"]["planVersion"] == 4
    assert payload["snapshot"]["planner"]["activeInstruction"] == "inspect loading dock"
    assert payload["snapshot"]["planner"]["recoveryState"] == "REPLAN_PENDING"
    assert payload["snapshot"]["transport"]["viewerPublish"] is False


def test_runtime_snapshot_builder_is_removed_in_favor_of_server_snapshot() -> None:
    assert not hasattr(AuraRuntimeCommandSource, "_build_runtime_snapshot")
