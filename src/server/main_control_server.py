from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from clients.worker_clients import (
    ExecutorLocomotionClient,
    PlanningSessionPlannerClient,
    SupervisorMemoryClient,
    SupervisorPerceptionClient,
    SupervisorTaskCommandClient,
)
from runtime.subgoal_executor import CommandEvaluation, SubgoalExecutor
from runtime.supervisor import Supervisor
from schemas.events import FrameEvent
from schemas.planning_context import PlanningContext
from schemas.world_state import WorldStateSnapshot

from .command_resolver import CommandResolver
from .decision_engine import DecisionEngine
from .planner_coordinator import PlannerCoordinator
from .safety_supervisor import SafetySupervisor
from .task_manager import TaskManager
from .world_state_store import WorldStateStore


@dataclass(frozen=True)
class ServerTickResult:
    planning_context: PlanningContext
    world_state: WorldStateSnapshot
    action_command: object | None
    command_vector: np.ndarray
    trajectory_update: object
    evaluation: CommandEvaluation
    status: object | None
    frame_header: object | None = None
    capture_report: dict[str, object] = field(default_factory=dict)
    sensor_meta: dict[str, object] = field(default_factory=dict)
    viewer_overlay: dict[str, object] = field(default_factory=dict)
    notices: tuple[object, ...] = ()


class MainControlServer:
    def __init__(
        self,
        args,
        *,
        supervisor: Supervisor,
        planning_session,
        executor: SubgoalExecutor,
    ) -> None:
        self._args = args
        self._perception_client = SupervisorPerceptionClient(supervisor)
        self._memory_client = SupervisorMemoryClient(supervisor)
        self._task_command_client = SupervisorTaskCommandClient(supervisor)
        self._planner_client = PlanningSessionPlannerClient(planning_session)
        self._planner_coordinator = PlannerCoordinator(
            perception_client=self._perception_client,
            memory_client=self._memory_client,
            locomotion_client=ExecutorLocomotionClient(executor),
        )
        self._task_manager = TaskManager(args)
        self._decision_engine = DecisionEngine()
        self._command_resolver = CommandResolver()
        self._safety_supervisor = SafetySupervisor(args)
        self._world_state = WorldStateStore(initial_mode=str(getattr(args, "planner_mode", "")))

    @property
    def task_manager(self) -> TaskManager:
        return self._task_manager

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._planner_coordinator.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._planner_coordinator.shutdown()

    def bootstrap(self) -> tuple[object, ...]:
        notices = tuple(
            self._task_manager.bootstrap(
                nav_client=self._planner_client,
                s2_client=self._planner_client,
                memory_client=self._memory_client,
            )
        )
        self._world_state.set_mode(self._task_manager.mode)
        self._world_state.update_task(self._task_manager.snapshot())
        return notices

    def submit_interactive_instruction(self, instruction: str, *, source: str, task_id: str = "") -> tuple[int, object]:
        command_id, notice = self._task_manager.submit_interactive_instruction(
            instruction,
            source=source,
            task_id=task_id,
            nav_client=self._planner_client,
            s2_client=self._planner_client,
            memory_client=self._memory_client,
        )
        self._world_state.update_task(self._task_manager.snapshot())
        return command_id, notice

    def cancel_interactive_task(self, *, source: str) -> tuple[bool, object | None]:
        cancelled, notice = self._task_manager.cancel_interactive_task(
            source=source,
            s2_client=self._planner_client,
            memory_client=self._memory_client,
        )
        self._world_state.update_task(self._task_manager.snapshot())
        return cancelled, notice

    def snapshot(self) -> WorldStateSnapshot:
        return self._world_state.snapshot()

    def tick(
        self,
        *,
        frame_event: FrameEvent,
        task_events: Sequence[object],
        runtime_status,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> ServerTickResult:
        notices = []
        for event in task_events:
            notices.extend(
                self._task_manager.handle_event(
                    event,
                    nav_client=self._planner_client,
                    s2_client=self._planner_client,
                    memory_client=self._memory_client,
                )
            )

        self._world_state.set_mode(self._task_manager.mode)
        self._world_state.update_task(self._task_manager.snapshot())
        self._world_state.ingest_frame(frame_event)

        active_memory_instruction = self._planner_client.active_memory_instruction()
        directive = self._decision_engine.evaluate(
            world_state=self._world_state.snapshot(),
            task=self._task_manager.snapshot(),
            frame_event=frame_event,
            manual_command_present=self._task_manager.manual_command() is not None,
            active_memory_instruction=active_memory_instruction,
        )

        observation, enriched_batch = self._planner_coordinator.enrich_observation(
            frame_event=frame_event,
            retrieve_memory=directive.retrieve_memory,
            instruction=active_memory_instruction,
        )
        self._world_state.record_perception(enriched_batch)
        self._world_state.record_memory_context(None if observation is None else observation.memory_context)

        task_command = None
        if directive.route_task_command:
            task_command = self._task_command_client.step(
                now=time.time(),
                robot_pose=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
                robot_yaw_rad=float(frame_event.robot_yaw_rad),
                action_status=runtime_status,
                publish=False,
            )
        proposal = self._command_resolver.resolve_action_command(
            manual_command=self._task_manager.manual_command(),
            task_command=task_command,
        )
        planning_context = self._planner_coordinator.build_planning_context(
            frame_event=frame_event,
            task=self._task_manager.snapshot(),
            instruction=active_memory_instruction,
            planner_mode=self._task_manager.mode,
            perception_summary=self._world_state.snapshot().last_perception_summary,
            memory_summary=self._world_state.snapshot().last_memory_context,
            manual_command=proposal.action_command,
        )

        execution = self._planner_coordinator.execute(
            frame_event=frame_event,
            observation=observation,
            action_command=proposal.action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32),
            robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
        )
        resolved = self._command_resolver.resolve_execution(
            proposal=proposal,
            execution=execution,
        )
        resolved = self._safety_supervisor.apply(
            frame_event=frame_event,
            resolved=resolved,
        )
        self._task_manager.sync_after_update(resolved.trajectory_update, memory_client=self._memory_client)
        self._world_state.update_task(self._task_manager.snapshot())
        self._world_state.record_planning_result(resolved.trajectory_update)
        self._world_state.record_command_decision(resolved)

        frame_header = None if enriched_batch is None else enriched_batch.frame_header
        capture_report = {} if enriched_batch is None else dict(enriched_batch.capture_report)
        viewer_overlay = {}
        if frame_header is not None:
            raw_overlay = frame_header.metadata.get("viewer_overlay", {})
            if isinstance(raw_overlay, dict):
                viewer_overlay = dict(raw_overlay)

        return ServerTickResult(
            planning_context=planning_context,
            world_state=self._world_state.snapshot(),
            action_command=resolved.action_command,
            command_vector=np.asarray(resolved.command_vector, dtype=np.float32).copy(),
            trajectory_update=resolved.trajectory_update,
            evaluation=resolved.evaluation,
            status=resolved.status,
            frame_header=frame_header,
            capture_report=capture_report,
            sensor_meta=dict(frame_event.sensor_meta),
            viewer_overlay=viewer_overlay,
            notices=tuple(notices),
        )
