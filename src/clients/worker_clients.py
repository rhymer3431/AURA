from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ipc.messages import ActionCommand, ActionStatus
from runtime.planning_session import ExecutionObservation, PlanningSession
from runtime.subgoal_executor import SubgoalExecutor
from runtime.supervisor import Supervisor
from schemas.workers import (
    LocomotionRequest,
    LocomotionResult,
    MemoryRequest,
    MemoryResult,
    NavRequest,
    NavResult,
    PerceptionRequest,
    PerceptionResult,
    build_error_result,
    inherit_worker_metadata,
)


class PerceptionClient(Protocol):
    def process(self, request: PerceptionRequest) -> PerceptionResult:
        ...


class MemoryClient(Protocol):
    def retrieve(self, request: MemoryRequest) -> MemoryResult:
        ...

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        ...

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
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
    def plan(self, request: NavRequest) -> NavResult:
        ...


class LocomotionClient(Protocol):
    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        ...

    def shutdown(self) -> None:
        ...

    def execute(self, request: LocomotionRequest) -> LocomotionResult:
        ...


class SupervisorPerceptionClient:
    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def process(self, request: PerceptionRequest) -> PerceptionResult:
        metadata = inherit_worker_metadata(request.metadata, source="supervisor.perception")
        if request.batch is None:
            return build_error_result(
                PerceptionResult,
                metadata=metadata,
                error="missing batch",
                summary=self._summary(),
            )
        try:
            batch = self._supervisor.process_frame(request.batch, publish=bool(request.publish))
        except Exception as exc:  # noqa: BLE001
            return build_error_result(
                PerceptionResult,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
                summary=self._summary(),
            )
        return PerceptionResult(
            metadata=metadata,
            batch=batch,
            summary=self._summary(),
        )

    def _summary(self) -> dict[str, object]:
        detector = self._supervisor.perception_pipeline.detector
        detector_report = detector.runtime_report
        return {
            "detector_backend": str(detector.info.backend_name),
            "detector_selected_reason": str(detector.info.selected_reason),
            "detector_ready": bool(detector_report.ready_for_inference) if detector_report is not None else False,
            "detector_runtime_report": {} if detector_report is None else detector_report.as_dict(),
        }


class SupervisorMemoryClient:
    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def retrieve(self, request: MemoryRequest) -> MemoryResult:
        metadata = inherit_worker_metadata(request.metadata, source="supervisor.memory")
        try:
            memory_context = self._supervisor.memory_service.build_memory_context(
                instruction=request.instruction,
                current_pose=request.current_pose,
            )
        except Exception as exc:  # noqa: BLE001
            return build_error_result(
                MemoryResult,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
                summary=self._summary(),
            )
        return MemoryResult(
            metadata=metadata,
            memory_context=memory_context,
            summary=self._summary(),
        )

    def set_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self._supervisor.memory_service.set_planner_task(**kwargs)

    def clear_planner_task(self, **kwargs) -> None:  # noqa: ANN003
        self._supervisor.memory_service.clear_planner_task(**kwargs)

    def _summary(self) -> dict[str, object]:
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


class PlanningSessionTransportClient:
    def __init__(self, planning_session: PlanningSession) -> None:
        self._planning_session = planning_session

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        self._planning_session.ensure_navdp_service_ready(context=context)

    def ensure_dual_service_ready(self, *, context: str) -> None:
        self._planning_session.ensure_dual_service_ready(context=context)

    def capture_observation(self, frame_id: int, *, env=None) -> ExecutionObservation | None:  # noqa: ANN001
        return self._planning_session.capture_observation(frame_id, env=env)


class PlanningSessionNavClient:
    def __init__(self, planning_session: PlanningSession, *, planner: Any | None = None) -> None:
        self._planning_session = planning_session
        self._planner = planner

    def bind_planner(self, planner: Any) -> None:
        self._planner = planner

    def plan(self, request: NavRequest) -> NavResult:
        metadata = inherit_worker_metadata(request.metadata, source="planning_session.nav")
        if request.observation is None:
            return build_error_result(NavResult, metadata=metadata, error="missing observation")
        if request.robot_pos_world is None or request.robot_quat_wxyz is None:
            return build_error_result(NavResult, metadata=metadata, error="missing robot pose context")
        planner = self._planner or self._planning_session
        planner_fn = getattr(planner, "plan_with_observation", None)
        if not callable(planner_fn):
            return build_error_result(NavResult, metadata=metadata, error="planner transport does not support plan_with_observation")
        try:
            update = planner_fn(
                request.observation,
                action_command=request.action_command,
                robot_pos_world=np.asarray(request.robot_pos_world, dtype=np.float32),
                robot_yaw=float(request.robot_yaw),
                robot_quat_wxyz=np.asarray(request.robot_quat_wxyz, dtype=np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            return build_error_result(
                NavResult,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
            )
        result_metadata = inherit_worker_metadata(
            metadata,
            source="planning_session.nav",
            plan_version=int(update.plan_version),
            goal_version=int(update.goal_version),
            traj_version=int(update.traj_version),
        )
        return NavResult(
            metadata=result_metadata,
            trajectory_update=update,
            trajectory_world=np.asarray(update.trajectory_world, dtype=np.float32).copy(),
            latency_ms=float(update.stats.latency_ms),
        )


class ExecutorLocomotionClient:
    def __init__(self, executor: SubgoalExecutor) -> None:
        self._executor = executor

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._executor.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._executor.shutdown()

    def execute(self, request: LocomotionRequest) -> LocomotionResult:
        metadata = inherit_worker_metadata(
            request.metadata,
            source="executor.locomotion",
            plan_version=None if request.trajectory_update is None else int(request.trajectory_update.plan_version),
            goal_version=None if request.trajectory_update is None else int(request.trajectory_update.goal_version),
            traj_version=None if request.trajectory_update is None else int(request.trajectory_update.traj_version),
        )
        if request.trajectory_update is None:
            return build_error_result(LocomotionResult, metadata=metadata, error="missing trajectory update")
        if any(
            value is None
            for value in (
                request.robot_pos_world,
                request.robot_lin_vel_world,
                request.robot_ang_vel_world,
                request.robot_quat_wxyz,
            )
        ):
            return build_error_result(LocomotionResult, metadata=metadata, error="missing locomotion state context")
        try:
            proposal = self._executor.step(
                frame_idx=int(request.frame_idx),
                observation=request.observation,
                action_command=request.action_command,
                trajectory_update=request.trajectory_update,
                robot_pos_world=np.asarray(request.robot_pos_world, dtype=np.float32),
                robot_lin_vel_world=np.asarray(request.robot_lin_vel_world, dtype=np.float32),
                robot_ang_vel_world=np.asarray(request.robot_ang_vel_world, dtype=np.float32),
                robot_yaw=float(request.robot_yaw),
                robot_quat_wxyz=np.asarray(request.robot_quat_wxyz, dtype=np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            return build_error_result(
                LocomotionResult,
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
            )
        return LocomotionResult(metadata=metadata, proposal=proposal)
