from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import FrameHeader
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from schemas.commands import ResolvedCommand
from schemas.events import FrameEvent, WorkerMetadata
from schemas.world_state import TaskSnapshot
from runtime.subgoal_executor import CommandEvaluation
from server.world_state_store import WorldStateStore


def test_world_state_store_tracks_frame_memory_and_plan_versions() -> None:
    store = WorldStateStore(initial_mode="interactive")
    frame_event = FrameEvent(
        metadata=WorkerMetadata(task_id="interactive", frame_id=3, timestamp_ns=time.time_ns(), source="test", timeout_ms=100),
        frame_id=3,
        timestamp_ns=time.time_ns(),
        source="test",
        robot_pose_xyz=(1.0, 2.0, 0.0),
        robot_yaw_rad=0.5,
        sim_time_s=1.5,
        observation=None,
        batch=None,
    )
    batch = IsaacObservationBatch(
        frame_header=FrameHeader(frame_id=3, timestamp_ns=time.time_ns(), source="test"),
        robot_pose_xyz=(1.0, 2.0, 0.0),
        robot_yaw_rad=0.5,
        sim_time_s=1.5,
        observations=[object(), object()],
        capture_report={"sensor": "ok"},
    )
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        plan_version=7,
        stats=PlannerStats(successful_calls=1, failed_calls=0, latency_ms=3.0, last_plan_step=3),
        source_frame_id=3,
        goal_version=4,
        traj_version=5,
        stale_sec=0.2,
    )
    resolved = ResolvedCommand(
        action_command=None,
        command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        trajectory_update=update,
        evaluation=CommandEvaluation(force_stop=False, goal_distance_m=1.0, yaw_error_rad=0.0, reached_goal=False),
        status=None,
        source="manual",
    )

    store.update_task(TaskSnapshot(task_id="interactive", instruction="dock", mode="interactive", state="active", command_id=9))
    store.ingest_frame(frame_event)
    store.record_perception(batch)
    store.record_memory_context(type("Bundle", (), {"instruction": "dock", "text_lines": [1, 2], "keyframes": [1], "crop_path": "", "latent_backend_hint": "stub"})())
    store.record_planning_result(update)
    store.record_command_decision(resolved)

    snapshot = store.snapshot()
    assert snapshot.current_task.task_id == "interactive"
    assert snapshot.robot_pose_xyz == (1.0, 2.0, 0.0)
    assert snapshot.last_perception_summary["detection_count"] == 2
    assert snapshot.last_memory_context["text_line_count"] == 2
    assert snapshot.active_nav_plan["plan_version"] == 7
    assert snapshot.last_s2_result["goal_version"] == 4
    assert snapshot.last_command_decision["source"] == "manual"
