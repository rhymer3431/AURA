from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningServiceResult:
    planning_context: object
    execution: object
    planning_evaluation: object


class PlanningService:
    def __init__(self, *, planner_coordinator, decision_engine) -> None:  # noqa: ANN001
        self._planner_coordinator = planner_coordinator
        self._decision_engine = decision_engine

    def plan(
        self,
        *,
        frame_event,
        world_state,
        task,
        recovery_state,
        now_ns: int,
        instruction: str,
        planner_mode: str,
        perception_summary: dict[str, object],
        memory_summary: dict[str, object],
        action_command,
        allow_planning: bool,
        skip_reason: str,
        robot_pos_world,
        robot_lin_vel_world,
        robot_ang_vel_world,
        robot_yaw: float,
        robot_quat_wxyz,
        observation,
    ) -> PlanningServiceResult:  # noqa: ANN001
        planning_context = self._planner_coordinator.build_planning_context(
            frame_event=frame_event,
            task=task,
            instruction=instruction,
            planner_mode=planner_mode,
            perception_summary=perception_summary,
            memory_summary=memory_summary,
            manual_command=action_command,
        )
        if allow_planning:
            execution = self._planner_coordinator.execute(
                frame_event=frame_event,
                observation=observation,
                action_command=action_command,
                robot_pos_world=robot_pos_world,
                robot_lin_vel_world=robot_lin_vel_world,
                robot_ang_vel_world=robot_ang_vel_world,
                robot_yaw=robot_yaw,
                robot_quat_wxyz=robot_quat_wxyz,
            )
        else:
            execution = self._planner_coordinator.skip_execution(
                frame_event=frame_event,
                action_command=action_command,
                reason=skip_reason,
            )
        planning_evaluation = self._decision_engine.evaluate_planning_outcome(
            world_state=world_state,
            task=task,
            recovery_state=recovery_state,
            trajectory_update=execution.trajectory_update,
            action_command=action_command,
            now_ns=now_ns,
        )
        return PlanningServiceResult(
            planning_context=planning_context,
            execution=execution,
            planning_evaluation=planning_evaluation,
        )

