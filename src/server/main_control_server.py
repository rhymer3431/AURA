from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from clients.worker_clients import (
    ExecutorLocomotionClient,
    SupervisorMemoryClient,
    SupervisorPerceptionClient,
    SupervisorTaskCommandClient,
)
from systems.transport.messages import TaskRequest
from locomotion.types import CommandEvaluation
from runtime.subgoal_executor import SubgoalExecutor
from runtime.supervisor import Supervisor
from schemas.events import FrameEvent
from schemas.planning_context import PlanningContext
from schemas.world_state import RuntimeStateSnapshot, WorldStateSnapshot
from runtime_pipeline.control import TickPipeline
from runtime_pipeline.enrichment import EnrichmentService
from runtime_pipeline.planning import PlanningService
from runtime_pipeline.safety import SafetyResolutionService
from runtime_pipeline.state import WorldStatePort

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
        self._planner_coordinator = PlannerCoordinator(
            args,
            planning_session=planning_session,
            perception_client=self._perception_client,
            memory_client=self._memory_client,
            locomotion_client=ExecutorLocomotionClient(executor),
        )
        self._task_manager = TaskManager(args)
        self._command_resolver = CommandResolver()
        self._safety_supervisor = SafetySupervisor(args)
        self._decision_engine = DecisionEngine(policy=self._safety_supervisor.policy)
        self._world_state = WorldStateStore(
            initial_mode="IDLE",
            runtime=RuntimeStateSnapshot(
                launch_mode=str(getattr(args, "resolved_launch_mode", "")),
                viewer_publish=bool(getattr(args, "viewer_publish", False)),
                native_viewer=str(getattr(args, "native_viewer", "off")),
                scene_preset=str(getattr(args, "scene_preset", "")),
                show_depth=bool(getattr(args, "show_depth", False)),
                memory_store=bool(getattr(args, "memory_store", True)),
                detection_enabled=not bool(getattr(args, "skip_detection", False)),
                control_endpoint=str(getattr(args, "viewer_control_endpoint", "")),
                telemetry_endpoint=str(getattr(args, "viewer_telemetry_endpoint", "")),
                shm_name=str(getattr(args, "viewer_shm_name", "")),
            ),
        )
        self._world_state_port = WorldStatePort(self._world_state)
        self._enrichment_service = EnrichmentService(
            planner_coordinator=self._planner_coordinator,
            world_state_port=self._world_state_port,
        )
        self._planning_service = PlanningService(
            planner_coordinator=self._planner_coordinator,
            decision_engine=self._decision_engine,
        )
        self._safety_resolution_service = SafetyResolutionService(
            task_command_client=self._task_command_client,
            command_resolver=self._command_resolver,
            safety_supervisor=self._safety_supervisor,
            task_manager=self._task_manager,
            world_state_port=self._world_state_port,
            memory_client=self._memory_client,
            planner_coordinator=self._planner_coordinator,
        )
        self._tick_pipeline = TickPipeline(
            task_manager=self._task_manager,
            planner_coordinator=self._planner_coordinator,
            decision_engine=self._decision_engine,
            world_state_port=self._world_state_port,
            enrichment_service=self._enrichment_service,
            planning_service=self._planning_service,
            safety_resolution_service=self._safety_resolution_service,
        )

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
                planner_coordinator=self._planner_coordinator,
                memory_client=self._memory_client,
            )
        )
        self._planner_coordinator.set_execution_mode(self._task_manager.mode)
        self._world_state_port.set_mode(self._task_manager.mode)
        self._world_state_port.update_task(self._task_manager.snapshot())
        self._world_state_port.seed_planning_state(
            mode=self._task_manager.mode,
            instruction=self._task_manager.snapshot().instruction,
            route_state=self._task_manager.route_state_seed(),
        )
        self._world_state_port.reset_recovery_state(entered_at_ns=time.time_ns(), reason="task_reset")
        return notices

    def submit_task_request(self, request: TaskRequest) -> tuple[object, ...]:
        notices = tuple(
            self._task_manager.handle_event(
                request,
                planner_coordinator=self._planner_coordinator,
                memory_client=self._memory_client,
            )
        )
        self._planner_coordinator.set_execution_mode(self._task_manager.mode)
        self._world_state_port.set_mode(self._task_manager.mode)
        self._world_state_port.update_task(self._task_manager.snapshot())
        self._world_state_port.seed_planning_state(
            mode=self._task_manager.mode,
            instruction=self._task_manager.snapshot().instruction,
            route_state=self._task_manager.route_state_seed(),
        )
        self._world_state_port.reset_recovery_state(entered_at_ns=time.time_ns(), reason="task_reset")
        return notices

    def set_idle(self, *, source: str) -> tuple[bool, object | None]:
        changed, notice = self._task_manager.set_idle(
            source=source,
            planner_coordinator=self._planner_coordinator,
            memory_client=self._memory_client,
        )
        self._planner_coordinator.set_execution_mode(self._task_manager.mode)
        self._world_state_port.set_mode(self._task_manager.mode)
        self._world_state_port.update_task(self._task_manager.snapshot())
        self._world_state_port.seed_planning_state(mode="IDLE", instruction="", route_state={})
        if changed:
            self._world_state_port.reset_recovery_state(entered_at_ns=time.time_ns(), reason="task_reset")
        return changed, notice

    def submit_interactive_instruction(self, instruction: str, *, source: str, task_id: str = "") -> tuple[int, object]:
        notices = self.submit_task_request(TaskRequest(command_text=instruction, task_id=task_id or "task"))
        command_id = -1
        notice = notices[-1] if notices else None
        return command_id, notice

    def cancel_interactive_task(self, *, source: str) -> tuple[bool, object | None]:
        return self.set_idle(source=source)

    def snapshot(self) -> WorldStateSnapshot:
        return self._world_state_port.snapshot()

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
        return self._tick_pipeline.run(
            frame_event=frame_event,
            task_events=task_events,
            runtime_status=runtime_status,
            robot_pos_world=robot_pos_world,
            robot_lin_vel_world=robot_lin_vel_world,
            robot_ang_vel_world=robot_ang_vel_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )
