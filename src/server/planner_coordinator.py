from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from clients.worker_clients import (
    LocomotionClient,
    MemoryClient,
    NavClient,
    PerceptionClient,
    PlanningSessionNavClient,
)
from ipc.messages import ActionCommand
from locomotion.types import CommandEvaluation
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from schemas.commands import LocomotionProposal
from schemas.events import FrameEvent
from schemas.planning_context import PlanningContext
from schemas.workers import (
    LocomotionRequest,
    MemoryRequest,
    NavRequest,
    PerceptionRequest,
    finalize_worker_result,
    stamp_worker_metadata,
)
from schemas.world_state import TaskSnapshot

from .planner_runtime_engine import PlannerRuntimeEngine
from .planner_runtime_state import PlannerRuntimeState


class PlannerCoordinator:
    def __init__(
        self,
        args,
        *,
        planning_session,
        perception_client: PerceptionClient,
        memory_client: MemoryClient,
        locomotion_client: LocomotionClient,
        nav_client: NavClient | None = None,
    ) -> None:
        self._args = args
        self._transport = planning_session
        self._perception_client = perception_client
        self._memory_client = memory_client
        self._locomotion_client = locomotion_client
        self._runtime_state = PlannerRuntimeState(mode=str(getattr(planning_session, "mode", getattr(args, "planner_mode", ""))))
        self._engine = None
        if all(hasattr(planning_session, attr) for attr in ("navdp_client", "pointgoal_planner", "nogoal_planner", "_intrinsic")):
            self._engine = PlannerRuntimeEngine(args, transport=planning_session, state=self._runtime_state)
        self._nav_client = nav_client or PlanningSessionNavClient(
            planning_session,
            planner=self._engine if self._engine is not None else planning_session,
        )

    @property
    def runtime_state(self) -> PlannerRuntimeState:
        return self._runtime_state

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._transport.initialize(simulation_app, stage)
        self._locomotion_client.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._locomotion_client.shutdown()
        self._transport.shutdown()

    def ensure_navdp_service_ready(self, *, context: str) -> None:
        try:
            self._transport.ensure_navdp_service_ready(
                context=context,
                launcher_processes=self._runtime_state.launcher_processes,
            )
        except TypeError:
            self._transport.ensure_navdp_service_ready(context=context)

    def ensure_dual_service_ready(self, *, context: str) -> None:
        try:
            self._transport.ensure_dual_service_ready(
                context=context,
                launcher_processes=self._runtime_state.launcher_processes,
            )
        except TypeError:
            self._transport.ensure_dual_service_ready(context=context)

    def activate_interactive_roaming(self, reason: str) -> bool:
        if self._engine is None:
            return True
        return self._engine.activate_interactive_roaming(reason)

    def start_dual_task(self, instruction: str) -> None:
        if self._engine is None:
            starter = getattr(self._transport, "start_dual_task", None)
            if callable(starter):
                starter(instruction)
                return
            raise RuntimeError("planner transport does not support dual startup")
        self._engine.start_dual_task(instruction)

    def submit_interactive_instruction(self, instruction: str) -> int:
        if self._engine is None:
            submitter = getattr(self._transport, "submit_interactive_instruction", None)
            if callable(submitter):
                return int(submitter(instruction))
            raise RuntimeError("planner transport does not support interactive instructions")
        return int(self._engine.submit_interactive_instruction(instruction))

    def cancel_interactive_task(self) -> bool:
        if self._engine is None:
            canceller = getattr(self._transport, "cancel_interactive_task", None)
            if callable(canceller):
                return bool(canceller())
            return False
        return bool(self._engine.cancel_interactive_task())

    def active_memory_instruction(self) -> str:
        if self._engine is None:
            getter = getattr(self._transport, "active_memory_instruction", None)
            if callable(getter):
                return str(getter())
            return ""
        return self._engine.active_memory_instruction()

    def viewer_overlay_state(self) -> dict[str, object]:
        if self._engine is None:
            getter = getattr(self._transport, "viewer_overlay_state", None)
            if callable(getter):
                state = getter()
                if isinstance(state, dict):
                    return dict(state)
            return {}
        return self._runtime_state.viewer_overlay_state()

    def enrich_observation(
        self,
        *,
        frame_event: FrameEvent,
        retrieve_memory: bool,
        instruction: str,
    ):
        observation = frame_event.observation
        perception_result = None
        memory_result = None
        if frame_event.batch is not None:
            perception_request = PerceptionRequest(
                metadata=stamp_worker_metadata(
                    frame_event=frame_event,
                    source="planner_coordinator.perception",
                    task_id=str(frame_event.metadata.task_id),
                ),
                batch=frame_event.batch,
                publish=bool(frame_event.publish_observation),
            )
            perception_result = finalize_worker_result(
                self._perception_client.process(perception_request),
                expected=perception_request.metadata,
                now_ns=time.time_ns(),
            )
        if retrieve_memory and observation is not None:
            memory_request = MemoryRequest(
                metadata=stamp_worker_metadata(
                    frame_event=frame_event,
                    source="planner_coordinator.memory",
                    task_id=str(frame_event.metadata.task_id),
                ),
                instruction=str(instruction),
                current_pose=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
            )
            memory_result = finalize_worker_result(
                self._memory_client.retrieve(memory_request),
                expected=memory_request.metadata,
                now_ns=time.time_ns(),
            )
            if memory_result.ok and memory_result.memory_context is not None:
                observation = replace(observation, memory_context=memory_result.memory_context)
        return observation, perception_result, memory_result

    def build_planning_context(
        self,
        *,
        frame_event: FrameEvent,
        task: TaskSnapshot,
        instruction: str,
        planner_mode: str,
        perception_summary: dict[str, object],
        memory_summary: dict[str, object],
        manual_command,
    ) -> PlanningContext:
        current_goal = None
        if manual_command is not None and manual_command.target_pose_xyz is not None:
            current_goal = tuple(float(v) for v in manual_command.target_pose_xyz[:3])
        return PlanningContext(
            metadata=frame_event.metadata,
            task=task,
            planner_mode=str(planner_mode),
            instruction=str(instruction),
            robot_pose_xyz=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
            robot_yaw_rad=float(frame_event.robot_yaw_rad),
            current_goal=current_goal,
            perception_summary=dict(perception_summary),
            memory_summary=dict(memory_summary),
            manual_command=manual_command,
        )

    def execute(
        self,
        *,
        frame_event: FrameEvent,
        observation,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ):
        trajectory_update = self._plan_trajectory(
            frame_event=frame_event,
            observation=observation,
            action_command=action_command,
            robot_pos_world=robot_pos_world,
            robot_yaw=robot_yaw,
            robot_quat_wxyz=robot_quat_wxyz,
        )
        locomotion_request = LocomotionRequest(
            metadata=stamp_worker_metadata(
                frame_event=frame_event,
                source="planner_coordinator.locomotion",
                task_id=self._task_id(action_command, frame_event),
                plan_version=int(trajectory_update.plan_version),
                goal_version=int(trajectory_update.goal_version),
                traj_version=int(trajectory_update.traj_version),
            ),
            frame_idx=int(frame_event.frame_id),
            observation=observation,
            action_command=action_command,
            trajectory_update=trajectory_update,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32).copy(),
            robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32).copy(),
            robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32).copy(),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32).copy(),
        )
        locomotion_result = finalize_worker_result(
            self._locomotion_client.execute(locomotion_request),
            expected=locomotion_request.metadata,
            now_ns=time.time_ns(),
            plan_version=int(trajectory_update.plan_version),
            goal_version=int(trajectory_update.goal_version),
            traj_version=int(trajectory_update.traj_version),
        )
        if locomotion_result.ok and locomotion_result.proposal is not None:
            return locomotion_result.proposal
        return self._fallback_proposal(
            trajectory_update=trajectory_update,
            action_command=action_command,
            error=str(locomotion_result.error or locomotion_result.discard_reason),
        )

    def skip_execution(
        self,
        *,
        frame_event: FrameEvent,
        action_command: ActionCommand | None,
        reason: str,
    ) -> LocomotionProposal:
        trajectory = self._runtime_state.trajectory
        return LocomotionProposal(
            command_vector=np.zeros(3, dtype=np.float32),
            trajectory_update=TrajectoryUpdate(
                trajectory_world=np.asarray(trajectory.trajectory_world, dtype=np.float32).copy(),
                plan_version=int(trajectory.plan_version),
                stats=PlannerStats(last_plan_step=int(frame_event.frame_id)),
                source_frame_id=int(frame_event.frame_id),
                action_command=action_command,
                stop=False,
                planner_control_mode=trajectory.planner_control_mode,
                planner_yaw_delta_rad=trajectory.planner_yaw_delta_rad,
                stale_sec=float(trajectory.stale_sec),
                goal_version=int(self._runtime_state.goal.goal_version),
                traj_version=int(self._runtime_state.goal.traj_version),
                used_cached_traj=bool(trajectory.used_cached_traj),
                sensor_meta=dict(frame_event.sensor_meta),
                interactive_phase=str(self._runtime_state.interactive.phase),
                interactive_command_id=int(self._runtime_state.interactive.active_command_id),
                interactive_instruction=str(self._runtime_state.interactive.active_instruction),
            ),
            evaluation=CommandEvaluation(
                force_stop=False,
                goal_distance_m=-1.0,
                yaw_error_rad=0.0,
                reached_goal=False,
            ),
            metadata={"recovery_skip_reason": str(reason)},
        )

    def _plan_trajectory(
        self,
        *,
        frame_event: FrameEvent,
        observation,
        action_command: ActionCommand | None,
        robot_pos_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ) -> TrajectoryUpdate:
        if observation is None:
            return TrajectoryUpdate(
                trajectory_world=np.asarray(self._runtime_state.trajectory.trajectory_world, dtype=np.float32).copy(),
                plan_version=int(self._runtime_state.trajectory.plan_version),
                stats=PlannerStats(failed_calls=1, last_error="sensor data unavailable", last_plan_step=int(frame_event.frame_id)),
                source_frame_id=int(frame_event.frame_id),
                action_command=action_command,
                stop=True,
            )
        if action_command is not None and action_command.action_type == "LOOK_AT":
            return TrajectoryUpdate(
                trajectory_world=np.zeros((0, 3), dtype=np.float32),
                plan_version=int(self._runtime_state.trajectory.plan_version),
                stats=PlannerStats(last_plan_step=int(frame_event.frame_id)),
                source_frame_id=int(frame_event.frame_id),
                action_command=action_command,
                stop=False,
            )
        nav_request = NavRequest(
            metadata=stamp_worker_metadata(
                frame_event=frame_event,
                source="planner_coordinator.nav",
                task_id=self._task_id(action_command, frame_event),
            ),
            observation=observation,
            action_command=action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32).copy(),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32).copy(),
        )
        nav_result = finalize_worker_result(
            self._nav_client.plan(nav_request),
            expected=nav_request.metadata,
            now_ns=time.time_ns(),
        )
        if nav_result.ok and nav_result.trajectory_update is not None:
            return nav_result.trajectory_update
        return TrajectoryUpdate(
            trajectory_world=np.asarray(self._runtime_state.trajectory.trajectory_world, dtype=np.float32).copy(),
            plan_version=int(self._runtime_state.trajectory.plan_version),
            stats=PlannerStats(
                failed_calls=1,
                last_error=str(nav_result.error or nav_result.discard_reason or "planner result unavailable"),
                last_plan_step=int(frame_event.frame_id),
            ),
            source_frame_id=int(frame_event.frame_id),
            action_command=action_command,
            stop=True,
        )

    @staticmethod
    def _task_id(action_command: ActionCommand | None, frame_event: FrameEvent) -> str:
        if action_command is not None and str(action_command.task_id).strip() != "":
            return str(action_command.task_id)
        return str(frame_event.metadata.task_id)

    @staticmethod
    def _fallback_proposal(
        *,
        trajectory_update: TrajectoryUpdate,
        action_command: ActionCommand | None,
        error: str = "",
    ) -> LocomotionProposal:
        metadata: dict[str, object] = {}
        if str(error).strip() != "":
            metadata["worker_error"] = str(error)
        return LocomotionProposal(
            command_vector=np.zeros(3, dtype=np.float32),
            trajectory_update=trajectory_update,
            evaluation=CommandEvaluation(
                force_stop=True,
                goal_distance_m=-1.0,
                yaw_error_rad=0.0,
                reached_goal=False,
            ),
            metadata=metadata,
        )
