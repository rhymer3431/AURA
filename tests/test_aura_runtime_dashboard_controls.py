from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionCommand, FrameHeader, RuntimeControlRequest, TaskRequest
from runtime.aura_runtime import AuraRuntimeCommandSource
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from runtime.subgoal_executor import CommandEvaluation


class _FakePlanningSession:
    def __init__(self) -> None:
        self.events: list[str] = []

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self.events.append(f"navdp:{context}")

    def ensure_dual_service_ready(self, *, context: str) -> None:
        self.events.append(f"dual:{context}")

    def submit_interactive_instruction(self, instruction: str) -> int:
        self.events.append(f"submit:{instruction}")
        return 7

    def submit_interactive_point_goal(self, goal_world_xy, *, label: str = "") -> int:  # noqa: ANN001
        goal_xy = np.asarray(goal_world_xy, dtype=np.float32).reshape(-1)
        self.events.append(f"submit_pointgoal:{float(goal_xy[0]):.3f},{float(goal_xy[1]):.3f}:{label}")
        return 11

    def cancel_interactive_task(self) -> bool:
        self.events.append("cancel")
        return True


class _FakeMemoryService:
    def __init__(self) -> None:
        self.spatial_store = SimpleNamespace(objects=[1, 2], places=[1])
        self.semantic_store = SimpleNamespace(list=lambda: ["rule-a", "rule-b"])
        self.keyframes = [1, 2, 3]
        self.scratchpad = SimpleNamespace(
            instruction="inspect loading dock",
            planner_mode="interactive",
            task_state="active",
            task_id="interactive",
            command_id=7,
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


def _build_source() -> tuple[AuraRuntimeCommandSource, _FakePlanningSession, _FakeMemoryService, list[dict[str, object]]]:
    planning_session = _FakePlanningSession()
    memory_service = _FakeMemoryService()
    notices: list[dict[str, object]] = []
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
    source._mode = "interactive"
    source._launch_mode = "headless"
    source._viewer_publish = False
    source._native_viewer = "off"
    source._runtime_io = object()
    source._executor = SimpleNamespace(planning_session=planning_session)
    source._supervisor = SimpleNamespace(
        memory_service=memory_service,
        perception_pipeline=SimpleNamespace(
            detector=SimpleNamespace(
                info=SimpleNamespace(backend_name="stub-detector", selected_reason="unit-test"),
                runtime_report=_FakeRuntimeReport(),
            )
        ),
    )
    source._last_frame_header = FrameHeader(
        frame_id=5,
        timestamp_ns=123,
        source="aura_runtime",
        width=96,
        height=96,
        camera_pose_xyz=(0.0, 0.0, 1.2),
        robot_pose_xyz=(1.0, 2.0, 0.0),
        robot_yaw_rad=0.25,
        sim_time_s=4.0,
        metadata={},
    )
    source._last_capture_report = {"sensor": "ok"}
    source._last_sensor_meta = {"room_id": "warehouse"}
    source._last_viewer_overlay = {"detections": [{"class_name": "apple"}], "trajectory_pixels": [[1, 2], [3, 4]]}
    source._pending_status = None
    source._active_command = ActionCommand(action_type="LOCAL_SEARCH")
    source._publish_notice = lambda *, level, notice, details=None: notices.append(  # type: ignore[method-assign]
        {"level": level, "notice": notice, "details": dict(details or {})}
    )
    return source, planning_session, memory_service, notices


def test_handle_task_request_routes_dashboard_instruction_into_interactive_planner() -> None:
    source, planning_session, memory_service, notices = _build_source()

    source._handle_task_request(TaskRequest(command_text="go to the loading dock", task_id="task-1"))

    assert planning_session.events == [
        "navdp:interactive task (dashboard)",
        "dual:interactive task (dashboard)",
        "submit:go to the loading dock",
    ]
    assert memory_service.set_calls == [
        {
            "instruction": "go to the loading dock",
            "planner_mode": "interactive",
            "task_state": "pending",
            "task_id": "task-1",
            "command_id": 7,
        }
    ]
    assert notices[-1]["notice"] == "interactive task queued"
    assert notices[-1]["details"]["taskId"] == "task-1"


def test_handle_task_request_routes_pointgoal_command_into_navdp_only() -> None:
    source, planning_session, memory_service, notices = _build_source()

    source._handle_task_request(TaskRequest(command_text="/pointgoal 1.25 -0.5", task_id="task-pg"))

    assert planning_session.events == [
        "navdp:interactive pointgoal (dashboard)",
        "submit_pointgoal:1.250,-0.500:/pointgoal 1.250 -0.500",
    ]
    assert memory_service.set_calls == [
        {
            "instruction": "/pointgoal 1.250 -0.500",
            "planner_mode": "interactive",
            "task_state": "pending",
            "task_id": "task-pg",
            "command_id": 11,
        }
    ]
    assert notices[-1]["notice"] == "interactive point goal queued"
    assert notices[-1]["details"]["goal"] == {"x": 1.25, "y": -0.5}


def test_handle_runtime_control_cancels_active_interactive_task() -> None:
    source, planning_session, memory_service, notices = _build_source()

    source._handle_runtime_control(RuntimeControlRequest(action="cancel_interactive_task"))

    assert planning_session.events == ["cancel"]
    assert memory_service.clear_calls == [
        {
            "task_state": "cancelled",
            "reason": "interactive task cancelled via dashboard",
        }
    ]
    assert [item["notice"] for item in notices] == ["interactive task cancelled"]


def test_build_runtime_snapshot_contains_dashboard_contract_groups() -> None:
    source, _, _, _ = _build_source()

    snapshot = source._build_runtime_snapshot(
        update=TrajectoryUpdate(
            trajectory_world=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            plan_version=4,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=12.0, last_plan_step=5),
            source_frame_id=5,
            stale_sec=0.3,
            goal_version=2,
            traj_version=3,
            planner_control_mode="trajectory",
            planner_yaw_delta_rad=0.05,
            interactive_phase="task_active",
            interactive_command_id=7,
            interactive_instruction="inspect loading dock",
        ),
        evaluation=CommandEvaluation(
            force_stop=False,
            goal_distance_m=1.5,
            yaw_error_rad=0.04,
            reached_goal=False,
        ),
    )

    assert set(snapshot) == {"modes", "planner", "sensor", "perception", "memory", "transport"}
    assert snapshot["modes"]["plannerMode"] == "interactive"
    assert snapshot["planner"]["planVersion"] == 4
    assert snapshot["planner"]["interactiveInstruction"] == "inspect loading dock"
    assert snapshot["sensor"]["frameId"] == 5
    assert snapshot["perception"]["detectionCount"] == 1
    assert snapshot["memory"]["objectCount"] == 2
    assert snapshot["transport"]["viewerPublish"] is False
