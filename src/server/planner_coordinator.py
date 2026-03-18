from __future__ import annotations

from dataclasses import replace

import numpy as np

from clients.worker_clients import LocomotionClient, MemoryClient, PerceptionClient
from schemas.events import FrameEvent
from schemas.planning_context import PlanningContext
from schemas.world_state import TaskSnapshot


class PlannerCoordinator:
    def __init__(
        self,
        *,
        perception_client: PerceptionClient,
        memory_client: MemoryClient,
        locomotion_client: LocomotionClient,
    ) -> None:
        self._perception_client = perception_client
        self._memory_client = memory_client
        self._locomotion_client = locomotion_client

    def initialize(self, simulation_app, stage) -> None:  # noqa: ANN001
        self._locomotion_client.initialize(simulation_app, stage)

    def shutdown(self) -> None:
        self._locomotion_client.shutdown()

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
        action_command,
        robot_pos_world: np.ndarray,
        robot_lin_vel_world: np.ndarray,
        robot_ang_vel_world: np.ndarray,
        robot_yaw: float,
        robot_quat_wxyz: np.ndarray,
    ):
        return self._locomotion_client.execute(
            frame_idx=int(frame_event.frame_id),
            observation=observation,
            action_command=action_command,
            robot_pos_world=np.asarray(robot_pos_world, dtype=np.float32),
            robot_lin_vel_world=np.asarray(robot_lin_vel_world, dtype=np.float32),
            robot_ang_vel_world=np.asarray(robot_ang_vel_world, dtype=np.float32),
            robot_yaw=float(robot_yaw),
            robot_quat_wxyz=np.asarray(robot_quat_wxyz, dtype=np.float32),
        )
