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
from schemas.events import FrameEvent, WorkerMetadata
from schemas.commands import LocomotionProposal
from schemas.world_state import TaskSnapshot
from server.planner_coordinator import PlannerCoordinator


class _PerceptionClient:
    def __init__(self) -> None:
        self.calls = 0

    def process_frame(self, batch, *, publish: bool = True):  # noqa: ANN001
        self.calls += 1
        return batch


class _MemoryClient:
    def build_memory_context(self, *, instruction: str, current_pose: tuple[float, float, float]):  # noqa: ARG002
        return type("Bundle", (), {"instruction": instruction, "text_lines": [1], "keyframes": [], "crop_path": "", "latent_backend_hint": "stub"})()


class _LocomotionClient:
    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        return None

    def shutdown(self) -> None:
        return None

    def execute(self, **kwargs):  # noqa: ANN003
        action_command = kwargs["action_command"]
        return LocomotionProposal(
            command_vector=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
            trajectory_update=kwargs["trajectory_update"],
            evaluation=CommandEvaluation(force_stop=False, goal_distance_m=1.0, yaw_error_rad=0.0, reached_goal=False),
        )


def test_planner_coordinator_builds_context_and_attaches_memory_before_execution() -> None:
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
    coordinator = PlannerCoordinator(
        type("Args", (), {"planner_mode": "interactive"})(),
        planning_session=planning_transport,
        perception_client=_PerceptionClient(),
        memory_client=_MemoryClient(),
        locomotion_client=_LocomotionClient(),
    )
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

    enriched_observation, enriched_batch = coordinator.enrich_observation(
        frame_event=frame_event,
        retrieve_memory=True,
        instruction="find dock",
    )
    context = coordinator.build_planning_context(
        frame_event=frame_event,
        task=TaskSnapshot(task_id="interactive", instruction="find dock", mode="interactive", state="active", command_id=7),
        instruction="find dock",
        planner_mode="interactive",
        perception_summary={"detection_count": 1},
        memory_summary={"text_line_count": 1},
        manual_command=ActionCommand(action_type="LOCAL_SEARCH", task_id="interactive"),
    )

    assert enriched_batch is not None
    assert enriched_observation is not None and enriched_observation.memory_context is not None
    assert context.task.task_id == "interactive"
    assert context.memory_summary["text_line_count"] == 1
