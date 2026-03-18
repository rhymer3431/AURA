from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from adapters.sensors.isaac_bridge_adapter import IsaacObservationBatch
from ipc.messages import ActionCommand, ActionStatus
from runtime.planning_session import ExecutionObservation, PlanningSession
from runtime.subgoal_executor import SubgoalExecutionResult, SubgoalExecutor
from runtime.supervisor import Supervisor


class PerceptionClient(Protocol):
    def process_frame(self, batch: IsaacObservationBatch, *, publish: bool = True) -> IsaacObservationBatch:
        ...


class MemoryClient(Protocol):
    def build_memory_context(self, *, instruction: str, current_pose: tuple[float, float, float]):
        ...

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        ...

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        ...

    def summary(self) -> dict[str, object]:
        ...


class TaskCommandClient(Protocol):
    def step(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float],
        robot_yaw_rad: float | None = None,
        action_status: ActionStatus | None = None,
        publish: bool = False,
    ) -> ActionCommand | None:
        ...


class NavClient(Protocol):
    def ensure_navdp_service_ready(self, *, context: str) -> None:
        ...

    def capture_observation(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        ...

    def active_memory_instruction(self) -> str:
        ...

    def viewer_overlay_state(self) -> dict[str, object]:
        ...


class S2Client(Protocol):
    def ensure_dual_service_ready(self, *, context: str) -> None:
        ...

    def start_dual_task(self, instruction: str) -> None:
        ...

    def submit_interactive_instruction(self, instruction: str) -> int:
        ...

    def cancel_interactive_task(self) -> bool:
        ...


class LocomotionClient(Protocol):
    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        ...

    def shutdown(self) -> None:
        ...

    def execute(
        self,
        *,
        frame_idx: int,
        observation: ExecutionObservation | None,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> SubgoalExecutionResult:
        ...


class SupervisorPerceptionClient:
    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def process_frame(self, batch: IsaacObservationBatch, *, publish: bool = True) -> IsaacObservationBatch:
        return self._supervisor.process_frame(batch, publish=publish)


class SupervisorMemoryClient:
    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def build_memory_context(self, *, instruction: str, current_pose: tuple[float, float, float]):
        return self._supervisor.memory_service.build_memory_context(
            instruction=instruction,
            current_pose=current_pose,
        )

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self._supervisor.memory_service.set_planner_task(**kwargs)

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self._supervisor.memory_service.clear_planner_task(**kwargs)

    def summary(self) -> dict[str, object]:
        scratchpad = self._supervisor.memory_service.scratchpad
        return {
            "object_count": len(self._supervisor.memory_service.spatial_store.objects),
            "place_count": len(self._supervisor.memory_service.spatial_store.places),
            "semantic_rule_count": len(self._supervisor.memory_service.semantic_store.list()),
            "keyframe_count": len(self._supervisor.memory_service.keyframes),
            "scratchpad": {
                "instruction": str(scratchpad.instruction),
                "planner_mode": str(scratchpad.planner_mode),
                "task_state": str(scratchpad.task_state),
                "task_id": str(scratchpad.task_id),
                "command_id": int(scratchpad.command_id),
                "goal_summary": str(scratchpad.goal_summary),
                "recent_hint": str(scratchpad.recent_hint),
                "next_priority": str(scratchpad.next_priority),
            },
        }


class SupervisorTaskCommandClient:
    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def step(
        self,
        *,
        now: float,
        robot_pose: tuple[float, float, float],
        robot_yaw_rad: float | None = None,
        action_status: ActionStatus | None = None,
        publish: bool = False,
    ) -> ActionCommand | None:
        return self._supervisor.step(
            now=now,
            robot_pose=robot_pose,
            robot_yaw_rad=robot_yaw_rad,
            action_status=action_status,
            publish=publish,
        )


class PlanningSessionPlannerClient:
    def __init__(self, planning_session: PlanningSession) -> None:
        self._planning_session = planning_session

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self._planning_session.ensure_navdp_service_ready(context=context)

    def ensure_dual_service_ready(self, *, context: str) -> None:
        self._planning_session.ensure_dual_service_ready(context=context)

    def start_dual_task(self, instruction: str) -> None:
        self._planning_session.start_dual_task(instruction)

    def submit_interactive_instruction(self, instruction: str) -> int:
        return int(self._planning_session.submit_interactive_instruction(instruction))

    def cancel_interactive_task(self) -> bool:
        return bool(self._planning_session.cancel_interactive_task())

    def capture_observation(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        return self._planning_session.capture_observation(frame_id, env=env)

    def active_memory_instruction(self) -> str:
        return self._planning_session.active_memory_instruction()

    def viewer_overlay_state(self) -> dict[str, object]:
        getter = getattr(self._planning_session, "viewer_overlay_state", None)
        if callable(getter):
            state = getter()
            if isinstance(state, dict):
                return dict(state)
        return {}


class ExecutorLocomotionClient:
    def __init__(self, executor: SubgoalExecutor) -> None:
        self._executor = executor

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._executor.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._executor.shutdown()

    def execute(
        self,
        *,
        frame_idx: int,
        observation: ExecutionObservation | None,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> SubgoalExecutionResult:
        return self._executor.step(
            frame_idx=frame_idx,
            observation=observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_lin_vel_world=robot_lin_vel_world,
            robot_ang_vel_world=robot_ang_vel_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )
