from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from locomotion.types import CommandEvaluation
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.events import FrameEvent, WorkerMetadata
from schemas.workers import (
    LocomotionResult,
    MemoryResult,
    NavResult,
    S2Result,
    finalize_worker_result,
    stamp_worker_metadata,
)


def _frame_event() -> FrameEvent:
    return FrameEvent(
        metadata=WorkerMetadata(
            trace_id="trace_test",
            task_id="task-1",
            frame_id=4,
            timestamp_ns=time.time_ns(),
            source="runtime",
            timeout_ms=50,
        ),
        frame_id=4,
        timestamp_ns=time.time_ns(),
        source="runtime",
        robot_pose_xyz=(0.0, 0.0, 0.0),
        robot_yaw_rad=0.0,
        sim_time_s=1.0,
    )


def test_stamp_worker_metadata_carries_full_envelope() -> None:
    frame_event = _frame_event()
    metadata = stamp_worker_metadata(
        frame_event=frame_event,
        source="planner.memory",
        plan_version=7,
        goal_version=8,
        traj_version=9,
    )

    assert metadata.trace_id == "trace_test"
    assert metadata.task_id == "task-1"
    assert metadata.frame_id == 4
    assert metadata.source == "planner.memory"
    assert metadata.timeout_ms == 50
    assert metadata.plan_version == 7
    assert metadata.goal_version == 8
    assert metadata.traj_version == 9


def test_finalize_worker_result_discards_frame_task_and_version_mismatch() -> None:
    expected = WorkerMetadata(task_id="task-1", frame_id=4, timeout_ms=100, plan_version=1, goal_version=2, traj_version=3)
    result = NavResult(
        metadata=WorkerMetadata(
            trace_id=str(expected.trace_id),
            task_id="task-2",
            frame_id=5,
            timestamp_ns=int(expected.timestamp_ns),
            source="nav",
            timeout_ms=100,
            plan_version=10,
            goal_version=20,
            traj_version=30,
        ),
        trajectory_update=TrajectoryUpdate(
            trajectory_world=np.zeros((0, 3), dtype=np.float32),
            plan_version=10,
            stats=PlannerStats(last_plan_step=4),
            source_frame_id=4,
        ),
    )

    rejected = finalize_worker_result(result, expected=expected, plan_version=1, goal_version=2, traj_version=3)

    assert rejected.ok is False
    assert rejected.status == "discarded"
    assert "task_mismatch" in rejected.discard_reason or "frame_mismatch" in rejected.discard_reason


def test_finalize_worker_result_times_out_consistently_across_worker_results() -> None:
    expired = WorkerMetadata(
        task_id="task-1",
        frame_id=4,
        timestamp_ns=time.time_ns() - 5_000_000,
        source="worker",
        timeout_ms=1,
    )

    results = [
        MemoryResult(metadata=expired, summary={"kind": "memory"}),
        S2Result(metadata=expired, mode="wait"),
        LocomotionResult(
            metadata=expired,
            proposal=LocomotionProposal(
                command_vector=np.zeros(3, dtype=np.float32),
                trajectory_update=TrajectoryUpdate(
                    trajectory_world=np.zeros((0, 3), dtype=np.float32),
                    plan_version=0,
                    stats=PlannerStats(last_plan_step=4),
                    source_frame_id=4,
                ),
                evaluation=CommandEvaluation(force_stop=True, goal_distance_m=-1.0, yaw_error_rad=0.0, reached_goal=False),
            ),
        ),
    ]

    finalized = [finalize_worker_result(result) for result in results]

    assert all(result.status == "timeout" for result in finalized)
