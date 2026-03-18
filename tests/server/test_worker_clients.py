from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients.worker_clients import (
    ExecutorLocomotionClient,
    PlanningSessionNavClient,
    SupervisorMemoryClient,
    SupervisorPerceptionClient,
)
from locomotion.types import CommandEvaluation
from runtime.planning_session import ExecutionObservation, PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.events import WorkerMetadata
from schemas.workers import LocomotionRequest, MemoryRequest, NavRequest, PerceptionRequest


class _DetectorReport:
    ready_for_inference = True

    def as_dict(self) -> dict[str, object]:
        return {"ready": True}


class _Supervisor:
    def __init__(self) -> None:
        self.perception_pipeline = type(
            "Pipeline",
            (),
            {
                "detector": type(
                    "Detector",
                    (),
                    {
                        "info": type("Info", (), {"backend_name": "mock", "selected_reason": "test"})(),
                        "runtime_report": _DetectorReport(),
                    },
                )(),
            },
        )()
        self.memory_service = type(
            "MemoryService",
            (),
            {
                "spatial_store": type("SpatialStore", (), {"objects": [1], "places": [1, 2]})(),
                "semantic_store": type("SemanticStore", (), {"list": lambda self: [1, 2, 3]})(),
                "keyframes": [1, 2],
                "scratchpad": type(
                    "Scratchpad",
                    (),
                    {
                        "instruction": "find dock",
                        "planner_mode": "interactive",
                        "task_state": "active",
                        "task_id": "task-1",
                        "command_id": 7,
                        "goal_summary": "dock",
                        "recent_hint": "",
                        "next_priority": "",
                    },
                )(),
            },
        )()

    def process_frame(self, batch, *, publish: bool = True):  # noqa: ANN001, ARG002
        return batch


def _memory_bundle(instruction: str):
    return type(
        "Bundle",
        (),
        {
            "instruction": instruction,
            "text_lines": [1],
            "keyframes": [],
            "crop_path": "",
            "latent_backend_hint": "stub",
        },
    )()


def test_supervisor_clients_return_typed_results() -> None:
    supervisor = _Supervisor()
    supervisor.memory_service.build_memory_context = lambda instruction, current_pose: _memory_bundle(instruction)  # type: ignore[attr-defined]

    perception_client = SupervisorPerceptionClient(supervisor)  # type: ignore[arg-type]
    memory_client = SupervisorMemoryClient(supervisor)  # type: ignore[arg-type]
    metadata = WorkerMetadata(task_id="task-1", frame_id=5, timestamp_ns=time.time_ns(), source="test", timeout_ms=100)

    perception_result = perception_client.process(
        PerceptionRequest(
            metadata=metadata,
            batch=type("Batch", (), {})(),
            publish=False,
        )
    )
    memory_result = memory_client.retrieve(
        MemoryRequest(
            metadata=metadata,
            instruction="find dock",
            current_pose=(0.0, 0.0, 0.0),
        )
    )

    assert perception_result.ok
    assert perception_result.batch is not None
    assert perception_result.summary["detector_backend"] == "mock"
    assert memory_result.ok
    assert memory_result.memory_context is not None
    assert memory_result.summary["object_count"] == 1


def test_nav_and_locomotion_clients_return_typed_results() -> None:
    update = TrajectoryUpdate(
        trajectory_world=np.asarray([[0.6, 0.0, 0.0]], dtype=np.float32),
        plan_version=3,
        stats=PlannerStats(successful_calls=1, last_plan_step=5),
        source_frame_id=5,
        goal_version=9,
        traj_version=10,
    )
    planner = type("Planner", (), {"plan_with_observation": lambda self, *args, **kwargs: update})()
    nav_client = PlanningSessionNavClient(planning_session=object(), planner=planner)  # type: ignore[arg-type]
    nav_result = nav_client.plan(
        NavRequest(
            metadata=WorkerMetadata(task_id="task-1", frame_id=5, timestamp_ns=time.time_ns(), source="test", timeout_ms=100),
            observation=ExecutionObservation(
                frame_id=5,
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth=np.ones((8, 8), dtype=np.float32),
                sensor_meta={},
                cam_pos=np.zeros(3, dtype=np.float32),
                cam_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                intrinsic=np.eye(3, dtype=np.float32),
            ),
            robot_pos_world=np.zeros(3, dtype=np.float32),
            robot_yaw=0.0,
            robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
    )

    executor = type(
        "Executor",
        (),
        {
            "initialize": lambda self, simulation_app, stage: None,
            "shutdown": lambda self: None,
            "step": lambda self, **kwargs: LocomotionProposal(
                command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
                trajectory_update=kwargs["trajectory_update"],
                evaluation=CommandEvaluation(force_stop=False, goal_distance_m=1.0, yaw_error_rad=0.0, reached_goal=False),
            ),
        },
    )()
    locomotion_client = ExecutorLocomotionClient(executor)  # type: ignore[arg-type]
    locomotion_result = locomotion_client.execute(
        LocomotionRequest(
            metadata=WorkerMetadata(
                task_id="task-1",
                frame_id=5,
                timestamp_ns=time.time_ns(),
                source="test",
                timeout_ms=100,
                plan_version=3,
                goal_version=9,
                traj_version=10,
            ),
            frame_idx=5,
            observation=None,
            action_command=None,
            trajectory_update=update,
            robot_pos_world=np.zeros(3, dtype=np.float32),
            robot_lin_vel_world=np.zeros(3, dtype=np.float32),
            robot_ang_vel_world=np.zeros(3, dtype=np.float32),
            robot_yaw=0.0,
            robot_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
    )

    assert nav_result.ok
    assert nav_result.trajectory_update is update
    assert nav_result.metadata.plan_version == 3
    assert locomotion_result.ok
    assert locomotion_result.proposal is not None
    assert locomotion_result.metadata.traj_version == 10
