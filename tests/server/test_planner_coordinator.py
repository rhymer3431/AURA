from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipc.messages import ActionCommand
from locomotion.types import CommandEvaluation
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.events import FrameEvent, WorkerMetadata
from schemas.workers import (
    LocomotionResult,
    MemoryResult,
    NavResult,
    PerceptionResult,
    inherit_worker_metadata,
)
from schemas.world_state import TaskSnapshot
from server.planner_coordinator import PlannerCoordinator


class _PerceptionClient:
    def __init__(self) -> None:
        self.calls = 0
        self.last_request = None

    def process(self, request):  # noqa: ANN001
        self.calls += 1
        self.last_request = request
        return PerceptionResult(
            metadata=inherit_worker_metadata(request.metadata, source="test.perception"),
            batch=request.batch,
            summary={"detection_count": 1},
        )


class _MemoryClient:
    def __init__(self) -> None:
        self.last_request = None

    def retrieve(self, request):  # noqa: ANN001
        self.last_request = request
        bundle = type(
            "Bundle",
            (),
            {
                "instruction": request.instruction,
                "text_lines": [1],
                "keyframes": [],
                "crop_path": "",
                "latent_backend_hint": "stub",
            },
        )()
        return MemoryResult(
            metadata=inherit_worker_metadata(request.metadata, source="test.memory"),
            memory_context=bundle,
            summary={"text_line_count": 1},
        )

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        _ = kwargs

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        _ = kwargs


class _NavClient:
    def __init__(self, *, frame_offset: int = 0) -> None:
        self.frame_offset = int(frame_offset)
        self.last_request = None

    def plan(self, request):  # noqa: ANN001
        self.last_request = request
        update = TrajectoryUpdate(
            trajectory_world=np.asarray([[0.8, 0.0, 0.0]], dtype=np.float32),
            plan_version=12,
            stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=1.5, last_plan_step=request.metadata.frame_id),
            source_frame_id=int(request.metadata.frame_id),
            action_command=request.action_command,
            stop=False,
            goal_version=4,
            traj_version=8,
        )
        metadata = WorkerMetadata(
            trace_id=str(request.metadata.trace_id),
            task_id=str(request.metadata.task_id),
            frame_id=int(request.metadata.frame_id + self.frame_offset),
            timestamp_ns=int(request.metadata.timestamp_ns),
            source="test.nav",
            timeout_ms=int(request.metadata.timeout_ms),
            plan_version=12,
            goal_version=4,
            traj_version=8,
        )
        return NavResult(
            metadata=metadata,
            trajectory_update=update,
            trajectory_world=np.asarray(update.trajectory_world, dtype=np.float32).copy(),
            latency_ms=1.5,
        )


class _LocomotionClient:
    def __init__(self) -> None:
        self.last_request = None

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        return None

    def shutdown(self) -> None:
        return None

    def execute(self, request):  # noqa: ANN001
        self.last_request = request
        return LocomotionResult(
            metadata=inherit_worker_metadata(
                request.metadata,
                source="test.locomotion",
                plan_version=request.metadata.plan_version,
                goal_version=request.metadata.goal_version,
                traj_version=request.metadata.traj_version,
            ),
            proposal=LocomotionProposal(
                command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
                trajectory_update=request.trajectory_update,
                evaluation=CommandEvaluation(force_stop=False, goal_distance_m=1.0, yaw_error_rad=0.0, reached_goal=False),
            ),
        )


def _frame_event() -> tuple[FrameEvent, ExecutionObservation]:
    observation = ExecutionObservation(
        frame_id=6,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        sensor_meta={"room_id": "dock"},
        cam_pos=np.zeros(3, dtype=np.float32),
        cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )
    frame_event = FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=6, timestamp_ns=time.time_ns(), source="test", timeout_ms=100),
        frame_id=6,
        timestamp_ns=time.time_ns(),
        source="test",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
        observation=observation,
        batch=type("Batch", (), {})(),
        sensor_meta={"room_id": "dock"},
    )
    return frame_event, observation


def _coordinator(*, nav_client=None) -> PlannerCoordinator:
    planning_transport = type(
        "PlanningTransport",
        (),
        {
            "mode": "interactive",
            "navdp_client": None,
            "pointgoal_planner": None,
            "nogoal_planner": None,
            "_intrinsic": np.eye(3, dtype=np.float32),
        },
    )()
    return PlannerCoordinator(
        type("Args", (), {"planner_mode": "interactive"})(),
        planning_session=planning_transport,
        perception_client=_PerceptionClient(),
        memory_client=_MemoryClient(),
        locomotion_client=_LocomotionClient(),
        nav_client=_NavClient() if nav_client is None else nav_client,
    )


def test_planner_coordinator_builds_context_and_attaches_memory_before_execution() -> None:
    coordinator = _coordinator()
    frame_event, _observation = _frame_event()

    enriched_observation, perception_result, memory_result = coordinator.enrich_observation(
        frame_event=frame_event,
        retrieve_memory=True,
        instruction="find dock",
    )
    context = coordinator.build_planning_context(
        frame_event=frame_event,
        task=TaskSnapshot(task_id="interactive", instruction="find dock", mode="interactive", state="active", command_id=7),
        instruction="find dock",
        planner_mode="interactive",
        perception_summary={} if perception_result is None else perception_result.summary,
        memory_summary={} if memory_result is None else memory_result.summary,
        manual_command=ActionCommand(action_type="LOCAL_SEARCH", task_id="interactive"),
    )

    assert perception_result is not None and perception_result.ok
    assert memory_result is not None and memory_result.ok
    assert enriched_observation is not None and enriched_observation.memory_context is not None
    assert context.task.task_id == "interactive"
    assert context.memory_summary["text_line_count"] == 1
    assert perception_result.metadata.task_id == "interactive"
    assert memory_result.metadata.frame_id == 6


def test_planner_coordinator_discards_mismatched_nav_results() -> None:
    coordinator = _coordinator(nav_client=_NavClient(frame_offset=1))
    frame_event, observation = _frame_event()

    execution = coordinator.execute(
        frame_event=frame_event,
        observation=observation,
        action_command=ActionCommand(action_type="LOCAL_SEARCH", task_id="interactive", metadata={"planner_managed": True}),
        robot_pos_world=np.zeros(3, dtype=np.float32),
        robot_lin_vel_world=np.zeros(3, dtype=np.float32),
        robot_ang_vel_world=np.zeros(3, dtype=np.float32),
        robot_yaw=0.0,
        robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert execution.trajectory_update.stats.failed_calls == 1
    assert "frame_mismatch" in execution.trajectory_update.stats.last_error
