from __future__ import annotations

from dataclasses import replace

import numpy as np

from clients.worker_clients import LocomotionClient, MemoryClient, PerceptionClient
from ipc.messages import ActionCommand
from runtime.planning_session import PlannerStats, TrajectoryUpdate
from schemas.events import FrameEvent
from schemas.planning_context import PlanningContext
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
        enriched_batch = frame_event.batch
        if enriched_batch is not None:
            enriched_batch = self._perception_client.process_frame(
                enriched_batch,
                publish=bool(frame_event.publish_observation),
            )
        if retrieve_memory and observation is not None:
            memory_context = self._memory_client.build_memory_context(
                instruction=str(instruction),
                current_pose=tuple(float(v) for v in frame_event.robot_pose_xyz[:3]),
            )
            observation = replace(observation, memory_context=memory_context)
        return observation, enriched_batch

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
        return self._locomotion_client.execute(
            frame_idx=int(frame_event.frame_id),
            observation=observation,
            action_command=action_command,
            trajectory_update=trajectory_update,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32),
            robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
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
        if self._engine is None:
            planner = getattr(self._transport, "plan_with_observation", None)
            if callable(planner):
                return planner(
                    observation,
                    action_command=action_command,
                    robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
                    robot_yaw=float(robot_yaw),
                    robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
                )
        return self._engine.plan_with_observation(
            observation,
            action_command=action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
        )
